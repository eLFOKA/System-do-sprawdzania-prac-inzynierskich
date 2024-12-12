import fitz
import re
import os
import json
import argparse

import language_tool_python

from APIzeroGPT import ZeroGPTClient

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
def get_args():
    parser = argparse.ArgumentParser(description="Sprawdzanie prac inżynierskich.")
    parser.add_argument("--pdf", type=str, required=True, help="Ścieżka do pliku PDF.")
    parser.add_argument("--headers", default="headers.json", type=str, help="Ścieżka do pliku JSON z nagłówkami.")
    parser.add_argument("--references", default="baza_prac", type=str, help="Ścieżka do folderu z pracami referencyjnymi.")
    parser.add_argument("--output", type=str, default="output.pdf", help="Nazwa pliku wyjściowego.")
    parser.add_argument("--login", default=None ,type=str, help="Login do API.")
    parser.add_argument("--password", default=None, type=str, help="Hasło do API.")
    parser.add_argument("--thrAI", default=85.0, type=float, help="Próg dla AI.")
    parser.add_argument("--thrPlag", default=85.0, type=float, help="Próg dla Plagiarism.")
    return parser.parse_args()

class ContentExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        """Ekstrakcja tekstu z PDF przy użyciu PyMuPDF."""
        document = fitz.open(self.file_path)
        structures = []
        order = 1

        for page_num in range(len(document)):
            page = document[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        x0, y0, x1, y1 = span["bbox"]
                        text = span["text"]
                        font = span["font"]  # Pobranie nazwy czcionki
                        font_size = round(span["size"])  # Pobranie rozmiaru czcionki

                        # Zaokrąglamy współrzędne do jedności
                        x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)

                        if text.strip():
                            structures.append({
                                "line_number": order,
                                "content": text.strip(),
                                "bounding_box": (x0, y0, x1, y1),
                                "page_number": page_num + 1,  # Dodanie numeru strony
                                "font": font,
                                "font_size": font_size,
                                "added_to_bloc": False
                            })
                            order += 1

        document.close()
        return structures
    
    def merge_lines_together(structures):
        i = 0
        while i < len(structures) - 1:  # Upewnij się, że nie wyjdziemy poza zakres listy
            x0, y0, x1, y1 = structures[i]["bounding_box"]
            x0_next, y0_next, x1_next, y1_next = structures[i + 1]["bounding_box"]

            # Sprawdź, czy linie są na tej samej wysokości (y0, y1) z małą toleranją
            if (y0 == y0_next or (y0 - 1 == y0_next or y0 + 1 == y0_next)) or y1 == y1_next:
                structures[i]["content"] += " " +  structures[i + 1]["content"]
                structures[i]["bounding_box"] = (x0, y0, x1_next, y1)
                del structures[i + 1]  # Usuń połączoną linię
            else:
                i += 1  # Przejdź do następnej linii tylko jeśli nie połączyliśmy bieżącej
        return structures

    def merge_text_to_paragraphs(structures, headers=None, no_pages=2):
        """
        Łączy linie w akapity na podstawie reguł:
        - Linie muszą być typu 'text'.
        - Linie muszą mieć taki sam odstęp między sobą (różnica w y0).
        - Nowy akapit zaczyna się, gdy x0 następnej linii jest większy niż poprzedniej lub odstęp różni się znacząco.
        """
        blocs_of_content = []  # Lista wynikowych bloków (akapity, nagłówki itp.)
        current_bloc = []  # Obecny akapit
        previous_y0 = None
        line_spacing = None
        in_bibitem = False  # Flaga informująca, czy jesteśmy w bloku `bibitem`

        for i, structure in enumerate(structures):
            if structure["page_number"] != no_pages:
                if structure["content"].isdigit():
                    blocs_of_content.append({
                        "block_type": "page_numeration",
                        "content": structure["content"],
                        "content_normalized": StructureNormalizer.normalize_text(structure["content"]),
                        "bounding_box": structure["bounding_box"],
                        "page_number": structure["page_number"]
                    })
                # Rozpoznanie nagłówka
                elif structure["content_normalized"] in headers:
                    #print(structure["content_normalized"])
                    blocs_of_content.append({
                        "block_type": "Header",
                        "content": structure["content"],
                        "content_normalized": StructureNormalizer.normalize_text(structure["content"]),
                        "bounding_box": structure["bounding_box"],
                        "page_number": structure["page_number"]
                    })
                # Jeśli ma postać "(1.1)" to znaczy, że to numeracja równań matematycznych
                elif re.match(r"\(\d+\.\d+\)", structure["content_normalized"]):
                    blocs_of_content.append({
                        "block_type": "equation_numbering",
                        "content": structure["content"],
                        "content_normalized": StructureNormalizer.normalize_text(structure["content"]),
                        "bounding_box": structure["bounding_box"],
                        "page_number": structure["page_number"]
                    })
                # Łączenie tego co pozostało czyli zwykłego tekstu
                else:
                    x0, y0, x1, y1 = structure["bounding_box"]

                    if current_bloc:
                        last_line = current_bloc[-1]
                        last_x0, last_y0, _, last_y1 = last_line["bounding_box"]
                        current_spacing = y0 - last_y1  # Odstęp między liniami

                        # Łączenie w blok na podstawie odstępu, oraz tej samej strony
                        if (current_spacing <= 10 and
                            structure["page_number"] == last_line["page_number"] and
                            (x0 == last_x0 or x0 < last_x0)):  # Odstęp między x0
                            current_bloc.append(structure)  # Dodanie linii do bieżącego bloku
                        else:
                            # Zapisanie bieżącego bloku jako paragraf
                            blocs_of_content.append({
                                "block_type": "paragraph",
                                "content": " ".join([line["content"] for line in current_bloc]),
                                "content_normalized": " ".join([line["content_normalized"] for line in current_bloc]),
                                "bounding_box": (
                                    min(line["bounding_box"][0] for line in current_bloc),  # Najmniejsze x0
                                    min(line["bounding_box"][1] for line in current_bloc),  # Najmniejsze y0
                                    max(line["bounding_box"][2] for line in current_bloc),  # Największe x1
                                    max(line["bounding_box"][3] for line in current_bloc),  # Największe y1
                                ),
                                "page_number": current_bloc[0]["page_number"]
                            })
                            current_bloc = [structure]  # Rozpoczęcie nowego bloku
                    else:
                        # Pierwsza linia nowego bloku
                        current_bloc.append(structure)

        return blocs_of_content
    
    @staticmethod
    def load_headers(file_path):
        """Ładuje nagłówki z pliku JSON z poprawnym kodowaniem."""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def find_Table_of_Contents(structures):
        """Znajduje spis treści w tekście."""
        # wcztanie nagłóków z pliku json do zmiennej predefined headers
        predefined_headers = ContentExtractor.load_headers("headers.json")
        headings = list(predefined_headers['headings'].values())
        table_of_contents = predefined_headers["headings"]["TableofContents"]

        first_header_after_table_of_contents = predefined_headers["Utilites"]["FirstHeaderAfterTableofContent"]

        first_header = None
        was_found = False
        headers = []
        for structure in structures:
            if structure["content_normalized"] == table_of_contents:
                structure["type"] = "table_of_contents"
                structure["added_to_bloc"] = True
                was_found = True
            elif was_found and structure["content_normalized"] and structure["content_normalized"][-1].isdigit():
                structure["type"] = "table_of_contents"
                structure["content_normalized"] = StructureNormalizer.normalize_table_of_contents(structure["content_normalized"])
                structure["added_to_bloc"] = True
                headers.append(structure["content_normalized"])
            elif was_found and structure["content_normalized"] == first_header_after_table_of_contents:
                break
            else:    
                continue

        headers = headings + headers

        return headers
    
    def remove_page_numbers(structures):
        """Usuwa numery stron z listy struktur."""
        return [structure for structure in structures if not structure["content"].isdigit()]

    def extract_svg(self):
        """
        Łączy rysunki na każdej stronie w grupy na podstawie przestrzennego rozmieszczenia
        i zwraca bounding boxy dla grup rysunków.

        Returns:
            list: Lista zawierająca informacje o grupach rysunków z każdej strony.
        """
        distance_threshold = 10

        def is_close(rect1, rect2, threshold):
            """Sprawdza, czy dwa bounding boxy są blisko siebie."""
            return (
                rect1.x1 + threshold >= rect2.x0 and
                rect2.x1 + threshold >= rect1.x0 and
                rect1.y1 + threshold >= rect2.y0 and
                rect2.y1 + threshold >= rect1.y0
            )

        doc = fitz.open(self.file_path)
        grouped_results = []  # Przechowywanie wyników grupowania rysunków

        for page_num, page in enumerate(doc, start=1):
            drawings = page.get_drawings()
            if not drawings:
                #print(f"Strona {page_num}: brak rysunków.")
                continue

            # Pobieranie bounding boxów dla każdego rysunku
            rects = [fitz.Rect(item["rect"]) for item in drawings if "rect" in item]
            groups = []  # Lista grup na stronie

            # Grupowanie rysunków
            for rect in rects:
                added = False
                for group in groups:
                    if any(is_close(rect, g, distance_threshold) for g in group):
                        group.append(rect)
                        added = True
                        break
                if not added:
                    groups.append([rect])  # Tworzenie nowej grupy

            # Dodawanie bounding boxów grup do wyników
            page_groups = []
            for group in groups:
                min_x = min(r.x0 for r in group)
                min_y = min(r.y0 for r in group)
                max_x = max(r.x1 for r in group)
                max_y = max(r.y1 for r in group)
                group_rect = fitz.Rect(min_x, min_y, max_x, max_y)

                page_groups.append({
                    "bounding_box": (min_x, min_y, max_x, max_y),
                    "page_number": page_num
                })

            if page_groups:
                grouped_results.extend(page_groups)

        doc.close()
        return grouped_results

    def extract_images(self):
        """
        Ekstrakcja obrazów z PDF z użyciem page.get_text("dict").
        
        Returns:
            list: Lista zawierająca informacje o obrazach (bounding box i numer strony).
        """
        doc = fitz.open(self.file_path)
        image_results = []  # Lista na wyniki

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            text_dict = page.get_text("dict")  # Pobranie pełnego tekstu jako słownik
            
            for block in text_dict.get("blocks", []):  # Iteracja przez bloki
                if "image" in block:  # Sprawdź, czy blok zawiera obraz
                    bbox = block["bbox"]  # Współrzędne obrazu

                    # Dodaj informacje o obrazie do wyników
                    image_results.append({
                        "bounding_box": tuple(bbox),
                        "page_number": page_number + 1
                    })

        doc.close()
        return image_results

    def load_reference_works(folder_path):
        """
        Wczytuje pliki PDF z folderu i wyciąga paragrafy.
        
        Args:
            folder_path (str): Ścieżka do folderu z pracami referencyjnymi.
        
        Returns:
            dict: Słownik z nazwami plików jako kluczami i listami paragrafów jako wartościami.
        """
        reference_paragraphs = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                print(f"Wczytywanie pliku: {filename}")
                extractor = ContentExtractor(file_path)

                strutures = extractor.extract_text()

                lines = ContentExtractor.merge_lines_together(strutures)
                # znormalizuj linie
                for line in lines:
                    line["content_normalized"] = StructureNormalizer.normalize_text(line["content"])
                    line["comment"] = None

                lines = ContentExtractor.remove_page_numbers(lines)
                headers = ContentExtractor.find_Table_of_Contents(lines)

                # Połącz linie w bloki
                blocs_of_content = ContentExtractor.merge_text_to_paragraphs(lines, headers)
                # normalizacja tekstu
                for bloc in blocs_of_content:
                    bloc["content_normalized"] = StructureNormalizer.normalize_text(bloc["content"])
                    bloc["content_normalized"] = StructureNormalizer.fix_hyphenation(bloc["content_normalized"])

                

                reference_paragraphs = [
                    {"filename": filename, "content_normalized": bloc["content_normalized"]}
                    for bloc in blocs_of_content if bloc["block_type"] == "paragraph"
                ]

        return reference_paragraphs

    def remove_bloc_from_table_of_content(blocs_of_content):
        """
        Usuwa wszystkie bloki znajdujące się na stronie zawierającej "SPIS TREŚCI".

        Args:
            blocs_of_content (list): Lista bloków tekstu.

        Returns:
            list: Zaktualizowana lista bloków tekstu.
        """
        # Znajdź numer strony, na której znajduje się spis treści
        table_of_contents_page = None
        for bloc in blocs_of_content:
            if bloc["block_type"] == "Header" and bloc["content_normalized"] == "SPIS TREŚCI":
                table_of_contents_page = bloc["page_number"]
                break

        if table_of_contents_page is None:
            # Jeśli nie znaleziono spisu treści, zwróć oryginalną listę
            return blocs_of_content
        else:
            # Tworzenie nowej listy z pominięciem bloków na stronie spisu treści
            blocs_of_content = [
                bloc for bloc in blocs_of_content if bloc["page_number"] != table_of_contents_page
            ]
            return blocs_of_content

class StructureNormalizer:
    @staticmethod
    def normalize_text(text):
        """Normalizuje tekst, usuwając akcenty, znaki specjalne i inne różnice."""
        import re
        import unicodedata

        # NFD: Rozbij znaki złożone na podstawowe + diakrytyki
        text = unicodedata.normalize('NFD', text)
        text = re.sub(r'ó', 'ó', text)  # Zastępuje `ó` na `ó`
        # Usuń diakrytyki (np. `ą` → `a`, `ę` → `e`)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Zamień specyficzne błędy w znakach
        text = re.sub(r'\s*˛\s*a', 'ą', text)  # Zamienia `˛` otoczone spacjami na `ą`
        text = re.sub(r'\s*´\s*s', 'ś', text)  # Zamienia `´\ns` na `ś`
        text = re.sub(r'\s*´\s*c', 'ć', text)  # Zamienia `´\nc` na `ć`
        text = re.sub(r'\s*˛\s*e', 'ę', text)  # Zamienia `˛\ne` na `ę`
        text = re.sub(r'\s*˙\s*z', 'ż', text)  # Zamienia `˙\nz` na `ż`
        text = re.sub(r'\s*´\s*z', 'ź', text)  # Zamienia `´\nz` na `ź`
        text = re.sub(r'\s*´\s*n', 'ń', text)  # Zamienia `´\nn` na `ń`
        text = re.sub(r'\s*´\s*l', 'ł', text)  # Zamienia `´\nl` na `ł`
        text = re.sub(r'\s*´\s*o', 'ó', text)  # Zamienia `´\no` na `ó`
        # zamień te błędy również dla znaków z dużymi literami
        text = re.sub(r'\s*˛\s*A', 'Ą', text)  # Zamienia `˛\nA` na `Ą`
        text = re.sub(r'\s*´\s*S', 'Ś', text)  # Zamienia `´\nS` na `Ś`
        text = re.sub(r'\s*´\s*C', 'Ć', text)  # Zamienia `´\nC` na `Ć`
        text = re.sub(r'\s*˛\s*E', 'Ę', text)  # Zamienia `˛\nE` na `Ę`
        text = re.sub(r'\s*˙\s*Z', 'Ż', text)  # Zamienia `˙\nZ` na `Ż`
        text = re.sub(r'\s*´\s*Z', 'Ź', text)  # Zamienia `´\nZ` na `Ź`
        text = re.sub(r'\s*´\s*N', 'Ń', text)  # Zamienia `´\nN` na `Ń`
        text = re.sub(r'\s*´\s*L', 'Ł', text)  # Zamienia `´\nL` na `Ł`
        text = re.sub(r'\s*´\s*O', 'Ó', text)  # Zamienia `´\nO` na `Ó`
        # Usunięcie nadmiarowych liter po `ą`, `ś`, `ć`, `ę`
        text = re.sub(r'ąa', 'ą', text)  # Zamienia `ąa` na `ą`
        text = re.sub(r'śs', 'ś', text)  # Zamienia `śs` na `ś`
        text = re.sub(r'ćc', 'ć', text)  # Zamienia `ćc` na `ć`
        text = re.sub(r'ęe', 'ę', text)  # Zamienia `ęe` na `ę`
        text = re.sub(r'żz', 'ż', text)  # Zamienia `żz` na `ż`
        text = re.sub(r'źz', 'ź', text)  # Zamienia `źz` na `ź`
        text = re.sub(r'ńn', 'ń', text)  # Zamienia `ńn` na `ń`
        text = re.sub(r'łl', 'ł', text)  # Zamienia `łl` na `ł`
        text = re.sub(r'óo', 'ó', text)  # Zamienia `óo` na `ó`
        text = re.sub(r'ĄA', 'Ą', text)  # Zamienia `ĄA` na `Ą`
        text = re.sub(r'ŚS', 'Ś', text)  # Zamienia `ŚS` na `Ś`
        text = re.sub(r'ĆC', 'Ć', text)  # Zamienia `ĆC` na `Ć`
        text = re.sub(r'ĘE', 'Ę', text)  # Zamienia `ĘE` na `Ę`
        text = re.sub(r'ŻZ', 'Ż', text)  # Zamienia `ŻZ` na `Ż`
        text = re.sub(r'ŹZ', 'Ź', text)  # Zamienia `ŹZ` na `Ź`
        text = re.sub(r'ŃN', 'Ń', text)  # Zamienia `ŃN` na `Ń`
        text = re.sub(r'ŁL', 'Ł', text)  # Zamienia `ŁL` na `Ł`
        text = re.sub(r'ÓO', 'Ó', text)  # Zamienia `ÓO` na `Ó`

        # Usuwanie nadmiarowych spacji
        text = re.sub(r'\s+', ' ', text)
    
        return text

    @staticmethod
    def fix_hyphenation(text):
        """Poprawianie przypadków 'edytor- skimi' → 'edytorskimi'."""
        # Dopasowanie wzorca: dowolny znak, myślnik, dowolne spacje, kolejny znak
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        return text

    @staticmethod
    def normalize_table_of_contents(text):
        """Normalizuje tekst spisu treści."""
        # Usuń ciągi kropek spacja-kropka oraz kropki na końcu
        text = re.sub(r'(\s\.\s)+|\.+$', '', text)
        # Usuń nadmiarowe spacje
        text = re.sub(r'\s+', ' ', text).strip()
        # Jeśli treść kończy się cyfrą, usuń ją
        text = re.sub(r'\s*\d+$', '', text).strip()
        # Usuń ciągi kropek w środku tekstu, ale zostaw struktury numerowe
        text = re.sub(r'(?<!\d)\.+(?!\d)', '', text)
        # Dodaj kropkę między cyframi, jeśli jej nie ma
        text = re.sub(r'(\d)(?=\d)', r'\1.', text)
        # Dodaj kropkę po samotnej cyfrze, jeśli jej nie ma
        text = re.sub(r'\b(\d)(?!\.)\b', r'\1.', text)
        return text

class CommentAdder:
    @staticmethod
    def add_error_to_line(lines, line_number, comment, page_number):
        """Dodaje komentarz obok konkretnego słowa w danej linii. na danej stronie."""
        for line in lines:
            if line["line_number"] == line_number and line["page_number"] == page_number:
                line["comment"] = f"Comment: {comment}"
                break

class ErrorChecker:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('pl')

    def check_errors(self, text):
        """Sprawdza błędy w tekście."""
        return self.tool.check(text)
    
    def check_captions_svg_objects(self, lines, svg_objects):
        """
        Sprawdza poprawność podpisów pod grafikami i tabelami.
        
        Args:
            lines (list): Lista linii tekstu.
            svg_objects (list): Lista obiektów SVG z ich bounding boxami i numerami stron.
        Returns:
            list: Zaktualizowana lista linii z dodanymi komentarzami.
        """
        for svg_object in svg_objects:
            bbox = svg_object["bounding_box"]
            page_number = svg_object["page_number"]

            # Znajdź linie tekstu pod grafiką
            line_below = None

            for line in lines:
                if line["page_number"] == page_number:
                    y0 = line["bounding_box"][1]

                    # Znajdź linię najbliżej dolnej krawędzi bounding boxu
                    if y0 >= bbox[3]:
                        if line_below is None or abs(y0 - bbox[3]) < abs(line_below["bounding_box"][1] - bbox[3]):
                            line_below = line

            # Sprawdź, czy linia poniżej zawiera słowo "Tabela"
            if line_below and "Tabela" in line_below["content"]:
                line_below["comment"] = "Błąd: Podpisy tabeli powinny znajdować się na górze."

        return lines
       
    def check_captions_image_objects(self, lines, image_objects):
        """
        Sprawdza poprawność podpisów dla obrazów.

        Args:
            lines (list): Lista linii tekstu.
            image_objects (list): Lista obrazów z ich bounding boxami i numerami stron.

        Returns:
            list: Zaktualizowana lista linii z dodanymi komentarzami.
        """
        for image in image_objects:
            bbox = image["bounding_box"]
            page_number = image["page_number"]

            # Znajdź linie tekstu nad obrazem
            line_above = None

            for line in lines:
                if line["page_number"] == page_number:
                    y1 = line["bounding_box"][3]

                    # Znajdź linię najbliżej górnej krawędzi bounding boxu
                    if y1 <= bbox[1]:  # Linia musi być nad obrazem
                        if line_above is None or abs(bbox[1] - y1) < abs(bbox[1] - line_above["bounding_box"][3]):
                            line_above = line

            # Sprawdź, czy linia powyżej zawiera słowo "Rysunek"
            if line_above and "Rysunek" in line_above["content"]:
                line_above["comment"] = "Błąd: Podpisy obrazów powinny znajdować się pod nimi."

        return lines

    @staticmethod
    def compare_paragraphs_with_references(text1, text2):
        """
        Porównuje dwa stringi tekstu i zwraca procentowy wynik podobieństwa.
        
        Args:
            text1 (str): Pierwszy string do porównania.
            text2 (str): Drugi string do porównania.
        
        Returns:
            float: Wynik podobieństwa w procentach (0.0 - 100.0).
        """
        vectorizer = TfidfVectorizer()
        
        # Tworzymy macierz TF-IDF dla obu tekstów
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Obliczamy podobieństwo cosinusowe
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        # Zwracamy wynik jako procent
        return similarity * 100

    def check_noText_between_headers(self, blocs_of_content):
        """
        Sprawdza, czy między nagłówkami nie ma tekstu.
        
        """
        for i in range(len(blocs_of_content) - 1):
            if blocs_of_content[i]["block_type"] == "Header":
                if blocs_of_content[i+1]["block_type"] == "Header":
                    if blocs_of_content[i]["page_number"] == blocs_of_content[i+1]["page_number"]:
                        blocs_of_content[i]["comment"] = "Błąd: Brak tekstu między nagłówkami."
                

        return blocs_of_content
                                        
class findContextInLines:
    @staticmethod
    def find_context_in_lines(lines, text):
        """
        Znajduje linię z największą liczbą wspólnych słów z podanym tekstem.
        
        Args:
            lines (list): Lista linii.
            text (str): Tekst do porównania.

        Returns:
            tuple: Numer linii i numer strony, jeśli znaleziono dopasowanie, w przeciwnym razie (None, None).
        """
        if text is None:
            print("Warning: 'text' is None. Skipping search.")
            return None, None

        # Podziel tekst na słowa
        words_in_text = set(re.findall(r'\b\w+\b', text))

        best_match = None
        best_score = 0

        for line in lines:
            line_words = set(re.findall(r'\b\w+\b', line["content"]))
            common_words = words_in_text.intersection(line_words)

            if len(common_words) > best_score:
                best_score = len(common_words)
                best_match = line

        if best_match:
            return best_match["line_number"], best_match["page_number"]

        return None, None

class AddCommentsToPDF:
    def __init__(self, pdf_path, lines, blocs_of_content):
        self.pdf_path = pdf_path
        self.lines = lines
        self.blocs_of_content = blocs_of_content

    def add_comments_to_pdf(self):
        doc = fitz.open(self.pdf_path)

        for line in self.lines:
            if line["comment"] is not None:
                comment_point = fitz.Point(line["bounding_box"][2], line["bounding_box"][3] - 20)
                page = doc[line["page_number"] - 1]
                page.add_text_annot(comment_point, line["comment"], icon="Comment")
                print(f"Komentarz dodany do strony {line['page_number']}, linia {line['line_number']}.")

        for bloc in self.blocs_of_content:
            if bloc["block_type"] == "Header" and bloc["comment"] is not None:
                    comment_point = fitz.Point(bloc["bounding_box"][2], bloc["bounding_box"][3] - 20)
                    page = doc[bloc["page_number"] - 1]
                    page.add_text_annot(comment_point, bloc["comment"], icon="Comment")
                    print(f"Komentarz dodany do strony {bloc['page_number']}.")


        for bloc in self.blocs_of_content:
            if bloc['AI_detection'] is not None:
                rect = fitz.Rect(bloc["bounding_box"])
                comment_point = fitz.Point(bloc["bounding_box"][2], bloc["bounding_box"][3] - 20)
                page = doc[bloc["page_number"] - 1]
                page.add_text_annot(comment_point, bloc["AI_detection"], icon="Comment")
                page.add_redact_annot(rect, text=bloc["AI_detection"])
                print(f"Komentarz dodany do strony {bloc['page_number']} ")

        for bloc in self.blocs_of_content:
            if bloc['Plagiarism'] is not None:
                rect = fitz.Rect(bloc["bounding_box"])
                comment_point = fitz.Point(bloc["bounding_box"][2], bloc["bounding_box"][3] - 20)
                page = doc[bloc["page_number"] - 1]
                page.add_text_annot(comment_point, bloc["Plagiarism"], icon="Comment")
                page.add_redact_annot(rect, text=bloc["Plagiarism"], text_color=(0, 0, 1))
                print(f"Komentarz dodany do strony {bloc['page_number']} ")



        doc.save("output.pdf")
        doc.close()

def remove_unnecessary(lines):

    for line in lines:
        if line["comment"] is not None and "To zdanie nie zaczyna się" in line["comment"]:
            line["comment"] = None
        elif line["comment"] is not None and "Brak niesparowanego symbolu" in line["comment"]:
            line["comment"] = None
    return lines

def load_pdf(args):
    file_path = args.pdf
    references_folder = args.references
    password = args.password
    login = args.login
    thresholdAI = args.thrAI
    thresholdPlagiarism = args.thrPlag


    if file_path:
        extractor = ContentExtractor(file_path)
        checker = ErrorChecker()

        structures = extractor.extract_text()

        lines = structures
        lines = ContentExtractor.merge_lines_together(structures)

        # normalizacja linii
        for line in lines:
            line["content_normalized"] = StructureNormalizer.normalize_text(line["content"])
            line["comment"] = None

        # nagłówki z pliku json
        lines = ContentExtractor.remove_page_numbers(lines)
        headers = ContentExtractor.find_Table_of_Contents(lines)

        # Połącz linie w bloki
        blocs_of_content = ContentExtractor.merge_text_to_paragraphs(lines, headers, no_pages=2)

        # normalizacja tekstu w blokach
        for bloc in blocs_of_content:
            bloc["content_normalized"] = StructureNormalizer.normalize_text(bloc["content"])
            bloc["content_normalized"] = StructureNormalizer.fix_hyphenation(bloc["content_normalized"])
            bloc["content"] = StructureNormalizer.fix_hyphenation(bloc["content"])
            bloc["comment"] = None
            bloc["Plagiarism"] = None
            bloc["AI_detection"] = None

        # normalizacja nagłówków
        for bloc in blocs_of_content:
            if bloc['block_type'] == 'Header':
                bloc["content_normalized"] = StructureNormalizer.normalize_table_of_contents(bloc["content_normalized"])

        # sprawdzanie czy między nagłówkami nie ma tekstu
        blocs_of_content = checker.check_noText_between_headers(blocs_of_content)

        # sprawdzanie czy umiejscowienie podpisów pod obrazami jest poprawne
        svg_objects = extractor.extract_svg()
        lines = checker.check_captions_svg_objects(lines, svg_objects)

        images = extractor.extract_images()
        lines = checker.check_captions_image_objects(lines, images)

        # Sprawdzanie podobieństwa do prac referencyjnych
        
        reference_paragraphs = ContentExtractor.load_reference_works(references_folder)

        for bloc in blocs_of_content:
            for ref_paragraph in reference_paragraphs:
                percenatage_similarity = checker.compare_paragraphs_with_references(
                    bloc["content_normalized"], ref_paragraph["content_normalized"]
                )

                if percenatage_similarity > thresholdPlagiarism:
                    if bloc["Plagiarism"] is None:
                        bloc["Plagiarism"] = (
                            f"Podobieństwo do pracy referencyjnej: {percenatage_similarity}%, "
                            f"z pracy {ref_paragraph['filename']}"
                        )
                        print(
                            f"Wykryto podobieństwo do pracy referencyjnej: {percenatage_similarity}%, "
                            f"z pracy {ref_paragraph['filename']}"
                        )


        # Sprawdzanie błędów pisowni
        for bloc in blocs_of_content:
            if bloc['block_type'] == 'paragraph' or bloc['block_type'] == 'Header':
                errors = checker.check_errors(bloc.get("content_normalized"))
                
                for error in errors:
                    if error.ruleIssueType == "grammar" or error.ruleIssueType == "typographical" or error.ruleIssueType == "style" or error.ruleIssueType == "duplication" or error.ruleIssueType == "inconsistency":
                        error_context = error.context

                        linia, strona = findContextInLines.find_context_in_lines(lines, error_context)
                        if linia != None:
                            CommentAdder.add_error_to_line(lines, linia, error.message, strona)

        # Sprawdzanie z zeroGPT
        if login is not None and password is not None:
            zeroGPT = ZeroGPTClient(login=login, password=password)

            for bloc in blocs_of_content:
                if bloc['block_type'] == 'paragraph':
                    if len(bloc['content_normalized']) > 250:
                        response = zeroGPT.analyze_text(bloc['content_normalized'])

                        # Jeśli odpowiedź jest None, pomiń ten blok
                        if response is None:
                            print("Brak odpowiedzi od ZeroGPT dla akapitu. Pomijanie...")
                            break
                        else:
                            fake_percentage = response['data']['fakePercentage']
                            if fake_percentage > thresholdAI:
                                bloc["AI_detection"] = f"System wykrył że tekst jest w : {response['data']['fakePercentage']}% wygenerowany przez AI"
                            
        else:
            print("Nie podano loginu i hasła do ZeroGPT. Pomijanie detekcji AI.")

        lines = remove_unnecessary(lines)
                
        # Dodawanie komentarzy do pliku PDF
        add_comments = AddCommentsToPDF(file_path, lines, blocs_of_content)
        add_comments.add_comments_to_pdf()   
        print("Zakończono przetwarzanie pliku PDF.")
    else:
        print("Nie podano ścieżki do pliku PDF.")

def save_to_file(paragraphs, filename):
        """Zapisuje akapity do pliku tekstowego."""
        with open(filename, "w", encoding="utf-8") as file:
            for paragraph in paragraphs:
                file.write(f"Block Type: {paragraph['block_type']}\n")
                file.write(f"Content (original): {paragraph['content']}\n")
                file.write(f"Content (normalized): {paragraph['content_normalized']}\n")
                file.write(f"Bounding Box: {paragraph['bounding_box']}\n")
                file.write("\n" + "=" * 50 + "\n")
        print(f"Zapisano do pliku: {filename}")

if __name__ == "__main__":
    
    arg = get_args()
    load_pdf(arg)
