import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

class WikipediaScraper:
    def __init__(self):
        self.link = {}
        self.figure = []
        self.images = []
        self.subtables = {}
        self.listcontent = []
        self.paragraphs = []
        self.headings = []
        self.paralink = []
        self.content = []
        self.is_chem = False

    def clean_html_tags(self, soup):
        try:
            # Remove unwanted tags like script, style, link that do not have any important text.
            for tag in soup.find_all(["script", "style", "link", "meta"]):
                try:
                    tag.decompose()
                except Exception as e:
                    print(f"Error decomposing tag: {e}")

            for tag in soup.find_all("sup"):
                # Remove unwanted reference superscripts
                try:
                    if not tag.get_text() or tag.get_text()[0] == "[":
                        tag.decompose()
                except Exception as e:
                    print(f"Error decomposing tag: {e}")
                else:
                    try:
                        # To detect charge in chemical formulae
                        tag.string = "``" + tag.string
                    except Exception:
                        # print(f"Error as tag string is empty")
                        pass
            # Remove empty paragraph at start of page
            for tag in soup.find_all("p", class_="mw-empty-elt"):
                try:
                    tag.decompose()
                except Exception as e:
                    print(f"Error decomposing empty element: {e}")
            return soup
        except Exception as e:
            print(f"Error cleaning HTML tags: {e}")
            return soup

    def fetch_wikipedia_content(self, url):
        try:
            # Fetch the page and parse it using Beautiful Soup.
            response = requests.get(url)

            return BeautifulSoup(response.text, "lxml")
        except Exception as e:
            print(f"Error fetching Wikipedia content: {e}")
            return None

    def find_chemical_formulae(self, text):
        try:
            # List of valid element symbols from the periodic table
            ELEMENTS = {
                "H",
                "He",
                "Li",
                "Be",
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Ge",
                "As",
                "Se",
                "Br",
                "Kr",
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Sb",
                "Te",
                "I",
                "Xe",
                "Cs",
                "Ba",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
                "Po",
                "At",
                "Rn",
                "Fr",
                "Ra",
                "Ac",
                "Th",
                "Pa",
                "U",
                "Np",
                "Pu",
                "Am",
                "Cm",
                "Bk",
                "Cf",
                "Es",
                "Fm",
                "Md",
                "No",
                "Lr",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds",
                "Rg",
                "Cn",
                "Nh",
                "Fl",
                "Mc",
                "Lv",
                "Ts",
                "Og",
            }
            # Regular expression to match potential chemical formulae
            pattern = r"[\[\(\{\s](?:{com}*[A-Z][a-z]?{com}*)+[\]\)\s\.,]|^(?:{com}*[A-Z][a-z]?{com}*)+[\]\)\s\.,]|[\[\(\{\s](?:{com}*[A-Z][a-z]?{com}*)+$|^(?:{com}*[A-Z][a-z]?{com}*)+$".replace("{com}",r'[\d\(\)\{\}\[\]+\u2212`]')
            matches = re.finditer(pattern, text)

            # Filter matches to only include valid chemical formula patterns
            chemical_formulae = []

            for matche in matches:
                match = matche.group()
                # print("match:",match)
                # Split the match into potential element-symbol/number sequences
                elements_in_match = re.findall(r"[A-Z][a-z]?|\d+|\[|\]|\(|\)|\{|\}", match)
                valid = True
                i = 0
                j = 0
                # Check if the extracted elements are valid symbols and follow the rules
                while i < len(elements_in_match):
                    if elements_in_match[i] in ELEMENTS:
                        j += len(elements_in_match[i])
                        i += 1
                    elif elements_in_match[i] in {"[", "]", "(", ")", "{", "}"}:
                        # Skip brackets
                        i += 1
                    elif elements_in_match[i].isdigit():
                        i += 1
                        j += 1
                    else:
                        valid = False
                        break
                if valid:
                    # Eliminate false positives
                    if "``" not in match and (j==1 or (j==2 and
                    (
                        "He" in match
                        or "At" in match
                        or "As" in match
                        or "No" in match
                        or "As" in match
                        or "In" in match
                        or "II" in match 
                    ))):
                        continue
                    chemical_formulae.append(matche.span())
            return chemical_formulae
        except Exception as e:
            print(f"Error in finding chemical formula: {e}")

    def process_chemical_formulae(self, chemical_formulae, text):
        try:
            # Covert common roman numerals to numbers
            text=text.replace("III"," 3 ").replace("II"," 2")
            offset = 0
            # Mapping for ambiguous element symbols
            element_names = {
                "Si": "Silicon",
                "Co": "Cobalt",
                "Bk": "Berkelium",
                "Sc": "Scandium",
                "Nh": "Nihonium",
                "Ho": "Holmium",
                "Cf": "Californium",
                "Os": "Osmium",
                "Sn": "Tin",
                "Cn": "Copernicium",
                "Po": "Polonium",
                "Bi": "Bismuth",
                "Yb": "Ytterbium",
                "No": "Nobelium",
                "In": "Indium",
                "Np": "Neptunium",
                "Pu": "Plutonium",
                "Hf": "Hafnium",
                "Ni": "Nickel",
                "Nb": "Niobium",
                "Bh": "Bohrium",
                "Cu": "Copper",
                "Cs": "Cesium",
                "Pb": "Lead",
                "Hs": "Hassium",
                "Sb": "Antimony"
            }
            for formu in chemical_formulae:
                formula = text[formu[0] + offset : formu[1] + offset]
                # print('Original:',formula)
                # Offset to adjust string length change
                off = 0
                i = 0
                while i + off < len(formula):
                    length = len(formula)
                    curr_index = i + off
                    c = formula[curr_index]
                    prefix = formula[:curr_index]
                    suffix = formula[curr_index + 1:]
                    if c == "[" or c == "{":
                        formula = prefix + "\nopen bracket\n" + suffix
                    elif c == "]" or c == "}":
                        formula = prefix + "\nclosed bracket\n" + suffix
                    elif c.isalpha():
                        # Substitute element names in case of ambiguity
                        if curr_index + 1 < length and self.is_chem and formula[curr_index + 1].islower() and formula[curr_index : curr_index + 2] in element_names:
                            formula = prefix + " " + formula[curr_index : curr_index + 2] + " " + element_names[formula[curr_index : curr_index + 2]] + " " + formula[curr_index + 2 :]
                            i+=1
                        else:
                            formula = prefix + " " + c + " " + suffix
                    elif c.isdigit():
                        j=i+1
                        # Add space before and after consecutive digits
                        while j+off<len(formula) and formula[j+off].isdigit():
                            j+=1
                        formula=prefix+" "+formula[curr_index:j+off]+" "+formula[j+off:]
                        i+=j-i-1
                    elif c == "+":
                        formula = prefix + "positive" + suffix
                    elif c == "\u2212":
                        formula = prefix + "negative" + suffix
                    elif c == "`" and curr_index + 1 < length and formula[curr_index+1]=="`":
                        # `` sign used in superscript processing
                        formula = prefix + " charge " + formula[curr_index + 2:]
                        i += 1
                    off += len(formula) - length
                    i += 1
                # print('Processed:',formula)
                text = text[: formu[0] + offset] + formula + text[formu[1] + offset :]
                offset += off
            text = text.replace("(I)", "1").replace("(aq)"," aqueous ").replace("(s)"," solid ").replace("(g)"," gaseous ").replace("(v)"," vapour ").replace("(l)"," liquid ").replace("``","")
            return text
        except Exception as e:
            print(f"Error in processing chemical formula: {e}")

    def process_chemistry(self,text):
        formuale = self.find_chemical_formulae(text)
        text = self.process_chemical_formulae(formuale,text)
        text = self.process_shortforms(text)
        return text

    def mhchem_to_tts(self,mhchem_notation):
        try:
            # Based on mhchem documentation
            # Mapping for ambiguous element symbols
            element_names = {
                "Si": "Silicon",
                "Co": "Cobalt",
                "Bk": "Berkelium",
                "Sc": "Scandium",
                "Nh": "Nihonium",
                "Ho": "Holmium",
                "Cf": "Californium",
                "Os": "Osmium",
                "Sn": "Tin",
                "Cn": "Copernicium",
                "Po": "Polonium",
                "Bi": "Bismuth",
                "Yb": "Ytterbium",
                "No": "Nobelium",
                "In": "Indium",
                "Np": "Neptunium",
                "Pu": "Plutonium",
                "Hf": "Hafnium",
                "Ni": "Nickel",
                "Nb": "Niobium",
                "Bh": "Bohrium",
                "Cu": "Copper",
                "Cs": "Cesium",
                "Pb": "Lead",
                "Hs": "Hassium",
                "Sb": "Antimony"
            }
            readable=mhchem_notation.replace("(s)"," solid ").replace("(aq)"," aqueous ").replace("(g)"," gaseous ").replace("(l)"," liquid ").replace(" v "," precipitate ").replace("(v)"," precipitate ").replace("(^)"," released as gas ").replace(" ^ "," released as gas ")
            # Define regex patterns for elements, isotopes, bonds, arrows, and charges
            
            # Matches consecutive elements and digits
            element_pattern = r'(\d*[A-Z][A-Za-z\d]+)'

            # Matches isotopes
            isotope_pattern = r'(?:\^\{?(\d+)\}?)(?:\_\{?\-?(\d+)\}?)([A-Z][a-z]?)([+-]?|\^\{[+-]\})'

            # Matches bonds (single, double, triple, etc.)
            bond_pattern = r'bond\{-\}|bond\{=\}|bond\{#\}|bond\{~\}'

            # Matches reaction arrows
            arrow_pattern = r'->|=>|<-|<=>'

            # Matches charges
            charge_pattern = r'\^(\d*[+\-]|\{\d*[+\-]\}|\{\d*\}[+\-])'

            # Matches charges that don't use ^ sign
            charge_pattern2 = r'([A-Z]|[a-z]|\d)([+\-])'

            # Matches exponents
            power_pattern = r'\^(\d+)'

            # Replacement function to format isotope notation in a readable format
            def replace_isotope(match):
                mass_num = match.group(1)   # Mass number
                atomic_num = match.group(2) # Atomic number
                element = match.group(3)    # Element symbol
                charge = "charge " + match.group(4).replace('+','positive').replace('-','negative') or ""   # Charge, if present
                
                # Format output with spaces between each component
                return f" {element} isotope with mass {mass_num} and atomic number {atomic_num}  {charge}".strip()
            
            # Step 0: Handle arrows, converting them to phrases like "yields", "reversible reaction"
            arrow_conversion = {'->': ' yields ', '=>': ' yields with condition ', '<-': ' reversible ', '<=>': ' is in equilibrium with '}
            readable = re.sub(arrow_pattern, lambda m: arrow_conversion[m.group(0)], readable)
            
            # Step 1: Handle isotopes.
            readable = re.sub(isotope_pattern, replace_isotope, readable)

            # Step 2: Handle charges
            readable = re.sub(charge_pattern, lambda m: f" charge {m.group(1).replace('{', '').replace('}', '').replace('+', '').replace('-', '')} {'positive' if '+' in m.group(1) else 'negative'}", readable)
            readable = re.sub(charge_pattern2, lambda m: f" {m.group(1)} charge {m.group(2).replace('+', '').replace('-', '')} {'positive' if '+' in m.group(2) else 'negative'}", readable)

            # Step 3: Convert elements, e.g., "Cl" to "C l" and element name in case of ambiguity
            readable = re.sub(element_pattern, lambda m: " ".join(m.group(1)) + (" " + element_names.get(m.group(1), "") if self.is_chem else ""), readable)

            # Step 4: Handle bonds, converting them to words like "single bond", "double bond"
            bond_conversion = {'bond{-}': ' single bond ', 'bond{=}': ' double bond ', 'bond{#}': ' triple bond ', 'bond{~}': ' resonance bond '}
            readable = re.sub(bond_pattern, lambda m: bond_conversion[m.group(0)], readable)

            # Step 5: Handle powers
            readable = re.sub(power_pattern, lambda m: f" raised to power {m.group(1)} ", readable)

            # To handle add/minus signs and overset above reaction arrow
            readable = readable.replace("+", " plus ").replace("-", " minus ").replace("yields [","->[")

            return readable
        except Exception as e:
            print(f"Error in mhchem_to_tts: {e}")

    def chem_in_math_element(self, latex):
        try:
            pattern = r"\\,|\\!|\\:|\\;|\\quad|\\enspace|\\qquad|\\hspace\{\dem\}"
            # Replace matches with a space
            latex = re.sub(pattern, " ", latex)
            # Find all \ce{...} in latex 
            ce=re.finditer(r"\\ce",latex)
            # List to store start and end of \ce
            indices=[]
            # List to store corresponding conversion to readable text
            converted=[]
            for c in ce:
                # Set chemistry page flag
                self.is_chem = True
                start=(c.end()+1)
                # Skip if last span includes current span
                if len(indices)>=1 and indices[-1][1]>start:
                    continue
                end=start
                bracket_count=1
                while bracket_count!=0:
                    end+=1
                    if latex[end]=="{":
                        bracket_count+=1
                    elif latex[end]=="}":
                        bracket_count-=1
                # print(latex[start+1:end])
                indices.append([start-3,end])
                converted.append(self.mhchem_to_tts(latex[start+1:end]))
            i=len(indices)-1
            # print(latex)
            while i!=-1:
                a=indices[i][0]
                b=indices[i][1]
                if "{\\" in latex[a+4:b] or "/" in latex[a+4:b]:
                    tag = "ce{"
                # Use \text if \ce doesn't contain any other delimitifier
                else:
                    tag = "text{"
                latex=latex[:a]+tag+converted[i]+latex[b:]
                i-=1            
            # print(latex)

            return latex

        except Exception as e:
            print(f"Error in chem_in_math: {e}")

    def replace_math_elements(self, soup):
        """Replace all math elements with a placeholder."""
        try:
            math_counter = 1
            for math_element in soup.find_all("span", class_="mwe-math-element"):
                try:
                    latex = math_element.img.get("alt")
                    # print(latex)
                    # Processing chemistry in \ce latex
                    latex=self.chem_in_math_element(latex)

                    # latex = latex.replace("/", "//")
                    url = "http://localhost:8080/api"
                    data = {"latexInput": latex}
                    response = requests.post(url, json=data)
                    text = "Math Equation"
                    if response.status_code == 200:  # 201 Created
                        text = response.json()["output"]
                        if self.is_chem:
                            text=text.replace("cap "," ").replace(" normal "," ")
                        # print(text)
                    else:
                        print(f"Error: {response.status_code}")
                        
                    math_element.clear()
                    math_element.string=text
                    math_counter += 1
                except Exception as e:
                    print(f"Error replacing math element: {e}")
            return soup
        except Exception as e:
            print(f"Error replacing math elements: {e}")
            return soup

    def process_chem2_su(self, soup):
        # For all chem2-su spans, reverse superscript and subscript order and add `` to indicate charge in superscript
        chemf=soup.find_all('span',class_='template-chem2-su')
        for chem in chemf:
            try:
                spans=chem.find_all('span')
                if len(spans)==2:
                    span1=spans[0].get_text()
                    spans[0].string=spans[1].get_text()
                    spans[1].string='``'+span1
            except Exception as e:
                print(f"Error in processing chem2-su: {e}")
    
    def process_dd_for_chem(self, soup):
        try:
            dds=soup.find_all('dd')
            for dd in dds:
                chemspan = dd.find_all("span", class_="chemf")
                for span in chemspan:
                    # Set chemistry page flag
                    self.is_chem = True
                    text = span.get_text()
                    text = (
                        text.replace("(s)", " solid ").replace("(aq)", " aqueous ").replace("(g)", " gaseous ").replace("(v)"," vapour ").replace("(l)"," liquid ")
                    )
                    text = self.process_chemistry(text)
                    # Replace arrow symbols with text
                    text = (
                        text.replace("\u2192", "yields")
                        .replace("\u2191", "released as gas")
                        .replace("\u2193", "as precipitate")
                        .replace("\u21cc", "is in equilibrium with")
                        .replace("\u21c4", "is in equilibrium with")
                    )
                    # print('dd:',text)
                    span.string=text

                # To handle dd tags without any span
                if not chemspan and not dd.find('span'):
                    text = dd.get_text()
                    text = self.process_chemistry(text)
                    text = (
                        text.replace("\u2192", "yields")
                        .replace("\u2191", "released as gas")
                        .replace("\u2193", "as precipitate")
                        .replace("\u21cc", "is in equilibrium with")
                        .replace("\u21c4", "is in equilibrium with")
                    )
                    dd.string=text
                
        except Exception as e:
            print(f"Error in dd_for_chem: {e}")

    def tableToString(self, tableList):
        try:
            tableString = ""
            cols = len(tableList[1])
            for rowCount in range(1, len(tableList)):
                tableString += f"Row {rowCount}: "
                for colCount in range(0, cols):
                    tableString += tableList[0][colCount] + ": " + tableList[rowCount][colCount] + ",  "
                tableString += "\n"
            return tableString
        except Exception as e:
            print(f"Error converting table to string: {e}")
            return ""

    def extract_links_images(self, element):
        try:
            for i in element.find_all("a"):
                try:
                    if isinstance(i.get_text(), str):
                        self.link[i.get_text()] = i.get("href")
                except Exception as e:
                    print(f"Error processing link: {e}")
            for i in element.find_all("img"):
                try:
                    self.images.append([i.get("src"), i.get("alt")])
                except Exception as e:
                    print(f"Error processing image: {e}")
        except Exception as e:
            print(f"Error extracting links and images: {e}")

    def process_table(self, element):
        try:
            # Used for processing the table elements.
            for i in element.find_all("a"):
                if type(i.get_text()) is str:
                    self.link[i.get_text()] = i.get("href")
            for i in element.find_all("img"):
                self.images.append([i.get("src"), i.get("alt")])

            rows = element.find_all("tr")
            cols = len(rows[0].find_all("th"))
            flg = False
            if cols == 1:
                if "vevent" in element["class"]:
                    cols = int(rows[0].find_all("th")[0]["colspan"])
                    flg = True
                else:
                    return ""

            if cols == 0:
                return ""

            table_data = []
            counts = [0] * cols
            textdata = [None] * cols
            rowCounter = 0
            allTableString = ""
            while rowCounter < len(rows):
                row = rows[rowCounter]
                cells = row.find_all(["th", "td"])
                if cells[0].name == "th" and int(cells[0].get("colspan", 1)) == cols:
                    if len(table_data) > 1:
                        self.listcontent.append(["table", pd.DataFrame(table_data)])
                        allTableString += pd.DataFrame(table_data).to_string(index=False, header=False) + "\n"
                    table_data.clear()
                    tableHeading = cells[0].get_text()
                    rowCounter += 1
                    tableList = []
                    while rowCounter < len(rows):
                        rowList = []
                        cells = rows[rowCounter].find_all(["th", "td"])
                        if cells[0].name == "th" and int(cells[0].get("colspan", 1)) == cols:
                            if flg:
                                if not tableList:
                                    allTableString += "\n"
                                    break
                                tableDf = pd.DataFrame(tableList)
                                tableString = tableDf.to_string(index=False, header=False) + "\n"
                                allTableString += tableHeading + "\n" + tableString + "\n"
                                self.subtables[tableHeading] = tableList
                                self.listcontent.append(["subtable " + tableHeading, tableDf])
                                table_data.clear()
                                tableList.clear()
                                tableHeading = cells[0].get_text()
                                rowCounter += 1
                                cells = rows[rowCounter].find_all(["th", "td"])
                            else:
                                break
                        for cel in cells:
                            rowList.append(cel.get_text(strip=True, separator=" "))
                        rowCounter += 1
                        tableList.append(rowList)
                    if not tableList:
                        allTableString += "\n"
                        continue
                    if not flg:
                        hdng = rows[0].find_all("th")
                        for cnt in range(0, len(hdng)):
                            hdng[cnt] = hdng[cnt].get_text(strip=True)
                        tableList = [hdng] + tableList
                    table_df = pd.DataFrame(tableList)
                    tableString = table_df.to_string(index=False, header=False) + "\n"
                    allTableString += tableHeading + "\n" + tableString + "\n"
                    self.subtables[tableHeading] = tableList
                    self.listcontent.append(["subtable " + tableHeading, table_df])
                    continue
                row_data = []
                cell_counter = 0
                current_column = 0
                while current_column < cols:
                    if counts[current_column] > 0:
                        row_data.append(textdata[current_column])
                        counts[current_column] -= 1
                        current_column += 1
                        continue
                    cell_text = cells[cell_counter].get_text(strip=True, separator=" ")
                    colspan = int(cells[cell_counter].get("colspan", 1))
                    rowspan = int(cells[cell_counter].get("rowspan", 1))
                    if rowspan > 1:
                        counts[current_column] = rowspan - 1
                        textdata[current_column] = cell_text
                    cell_counter += 1
                    current_column += colspan
                    row_data.append(cell_text)
                table_data.append(row_data)
                rowCounter += 1
            if not table_data:
                return allTableString
            table_df = pd.DataFrame(table_data)
            if flg:
                self.listcontent.append(["subtable", table_df])
            else:
                self.listcontent.append(["table", table_df])
            tableString = table_df.to_string(index=False, header=False)
            allTableString += tableString
            return allTableString
        except Exception as e:
            print(f"Error processing table: {e}")
            return ""

    def process_shortforms(self, text):
        try:
            # Regular expression to match shortforms
            pattern = r"(?:(?<=\s|\(|\[)|(?<=^))([A-Z]+)(?:(?=\s|\)|\]|[\.,])|(?=$))"
            expanded_text = re.sub(pattern, lambda m: " ".join(m.group(1)), text)
            return expanded_text

        except Exception as e:
            print(f"Error processing shortforms: {e}")


    def process_structure(self, soup):
        try:
            self.content = []
            for element in soup.find_all(
                ["div", "p", "table", "ul", "cite", "figure","dl"],recursive=False
            ):
                try:
                    if element.name=="div":
                        if "mw-heading" in element.get("class",[]):
                            children=element.find_all()
                            if not children or children[0].get("id","") in ["Bibliography", "External_links", "References"]:
                                break
                            child=children[0]
                            self.extract_links_images(child)
                            text = child.get_text(strip=True)
                            if text == "Contents":
                                continue
                            text = self.process_chemistry(text)
                            self.content.append(f"\n{text}\n")
                            self.headings.append(text)
                            self.listcontent.append([element.name, text])

                    elif element.name == "p":
                            self.extract_links_images(element)
                            self.process_paragraph(element)
                            
                    elif element.name == "dl":
                            self.extract_links_images(element)
                            self.process_dl(element)

                    elif element.name == "table":
                        table_str = self.process_table(element)
                        if table_str:
                            self.content.append("\n" + table_str + "\n")

                    elif element.name == "ul":
                        self.extract_links_images(element)
                        self.process_ul(element)

                    elif element.name == "figure":
                        if element.img.get("src"):
                            self.figure.append([element.img.get("src"), element.figcaption.get_text()])

                except Exception as inner_e:
                    print(f"Error processing element: {inner_e}")
        except Exception as e:
            print(f"Error in print_structure: {e}")

    def process_ul(self, element):
        if "mw-gallery-traditional" in element.get("class", []):
            flag = False
            for item in element.find_all("div", ["thumb", "gallerytext"]):
                if "thumb" in item.get("class", []):
                    self.content.append(item.find_all("img")[0].get("alt", ""))
                else:
                    self.content.append(item.get_text())
                    if flag:
                        self.listcontent[-1][1] += "\n" + item.get_text()
                    else:
                        self.listcontent.append([element.name, item.get_text()])
                        flag = True
        else:
            if element.find_parent("table", class_="vevent"):
                return
            flag = False
            for item in element.find_all("li"):
                self.content.append(item.get_text())
                if flag:
                    self.listcontent[-1][1] += "\n" + item.get_text()
                else:
                    self.listcontent.append([element.name, item.get_text()])
                    flag = True
            self.content.append("\n")

    def process_paragraph(self, element):
        temp = {}
        text = element.get_text()
        text = self.process_chemistry(text)
        for a in element.find_all("a"):
            if isinstance(a.get_text(), str):
                temp[a.get_text()] = a.get("href")
        self.paralink.append(temp)
        self.extract_links_images(element)
        self.content.append(text)
        self.paragraphs.append(text)
        self.listcontent.append([element.name, text])
        for cite in element.find_all("cite"):
            cite.decompose()
        return
    
    def process_dl(self, element):
        temp = {}
        text = element.get_text()
        text = self.process_chemistry(text)
        for a in element.find_all("a"):
            if isinstance(a.get_text(), str):
                temp[a.get_text()] = a.get("href")
        self.paralink.append(temp)
        self.extract_links_images(element)
        self.content.append(text)
        self.paragraphs.append(text)
        self.listcontent.append([element.name, text])
        for cite in element.find_all("cite"):
            cite.decompose()
        for repeat in element.find_all("dl"):
            repeat.decompose()
        # print(element)
        return

    def fetch_short_description(self, soup):
        try:
            short_desc_div = soup.find("div", class_="shortdescription")
            if short_desc_div:
                return short_desc_div.get_text(strip=True)
            return "No short description available."
        except Exception as e:
            print(f"Error fetching short description: {e}")
            return "No short description available."

    def print_structure(self):
        for i in self.content:
            print(i)

    def main(self, url):
        try:
            soup = self.fetch_wikipedia_content(url)
            if soup is None:
                return
            contentsoup = soup.find("div", id="mw-content-text")
            contentsoup = soup.find("div", class_="mw-content-ltr")
            soup = self.clean_html_tags(contentsoup)
            soup = self.replace_math_elements(soup)
            self.process_chem2_su(soup)
            self.process_dd_for_chem(soup)
            self.process_structure(soup)
            self.print_structure()
            # for i in self.images:
                # print(i)
        except Exception as e:
            print(f"Error in main: {e}")


if __name__ == "__main__":
    # url = "https://en.wikipedia.org/wiki/James_Bond"
    # url = "https://en.wikipedia.org/wiki/Equation"
    url = "https://en.wikipedia.org/wiki/Chemical_equation"
    scraper = WikipediaScraper()
    scraper.main(url)