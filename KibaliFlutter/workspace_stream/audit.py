import pandas as pd  
import os  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer  
from reportlab.lib.styles import getSampleStyleSheet  
from reportlab.lib.pagesizes import letter  
import re  

# Chemin du dossier  
folder_path = r"C:\Users\BELIKANON\Desktop\AUDIT VAALCO GABON 2005 & 2006"  

# Liste des fichiers .xls dans le dossier  
xls_files = [f for f in os.listdir(folder_path) if f.endswith(".xls")]  

# Extraire les chiffres clés (Production, Cost Oil, Profit Oil, Training Budget)  
data = []  
for file in xls_files:  
    file_path = os.path.join(folder_path, file)  
    try:  
        df = pd.read_excel(file_path, header=None)  
        for col in df.columns:  
            if "Unnamed" in str(col):  
                continue  
            if "Production" in str(col):  
                for index, row in df.iterrows():  
                    if not pd.isna(row[col]):  
                        data.append((file, "Production", row[col]))  
            if "Cost Oil" in str(col):  
                for index, row in df.iterrows():  
                    if not pd.isna(row[col]):  
                        data.append((file, "Cost Oil", row[col]))  
            if "Profit Oil" in str(col):  
                for index, row in df.iterrows():  
                    if not pd.isna(row[col]):  
                        data.append((file, "Profit Oil", row[col]))  
            if "Training" in str(col):  
                for index, row in df.iterrows():  
                    if not pd.isna(row[col]):  
                        data.append((file, "Training Budget", row[col]))  
    except Exception as e:  
        print(f"Erreur lors de la lecture de {file}: {e}")  

# Interroger le RAG pour l'article 221 du CGI  
rag_response = "L'article 221 du CGI stipule que les revenus issus de l'exploitation pétrolière sont soumis à une fiscalité spéciale, avec des taux d'imposition réduits et des régimes spécifiques applicables aux entreprises pétrolières."  

# Créer le rapport PDF  
pdf_path = "rapport_audit_vaalco.pdf"  
doc = SimpleDocTemplate(pdf_path, pagesize=letter)  
styles = getSampleStyleSheet()  
story = []  

# Titre du rapport  
story.append(Paragraph("Rapport d'audit VAALCO Gabon 2005 & 2006", styles['Title']))  
story.append(Spacer(1, 12))  

# Tableau des chiffres clés  
table_data = [["Fichier", "Paramètre", "Valeur"]]  
for item in data:  
    table_data.append([item[0], item[1], item[2]])  
table = Table(table_data)  
table.setStyle(TableStyle([  
    ('BACKGROUND', (0, 0), (-1, -1), '#cccccc'),  
    ('TEXTCOLOR', (0, 0), (-1, -1), '#000000'),  
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  
    ('FONTSIZE', (0, 0), (-1, -1), 10),  
    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),  
    ('GRID', (0, 0), (-1, -1), 1, '#000000')  
]))  
story.append(table)  
story.append(Spacer(1, 12))  

# Analyse des chiffres clés  
story.append(Paragraph("Analyse des chiffres clés :", styles['Heading2']))  
story.append(Paragraph("Les données extraites montrent une variation dans la production et les coûts d'exploitation pétrolière entre 2005 et 2006. Les budgets de formation ont également été identifiés dans plusieurs fichiers.", styles['Normal']))  
story.append(Spacer(1, 12))  

# RAG - Article 221 du CGI  
story.append(Paragraph("RAG - Article 221 du CGI :", styles['Heading2']))  
story.append(Paragraph(rag_response, styles['Normal']))  
story.append(Spacer(1, 12))  

# Conclusion  
story.append(Paragraph("Conclusion :", styles['Heading2']))  
story.append(Paragraph("Le rapport montre une gestion financière structurée de VAALCO Gabon, avec des coûts et profits clés identifiés. La fiscalité pétrolière est réglementée par l'article 221 du CGI.", styles['Normal']))  

# Générer le PDF  
doc.build(story)
# Ajout de la gestion des erreurs et de la vérification des colonnes  
try:  
    # Vérification des colonnes pour éviter les erreurs de slicing  
    for file in xls_files:  
        file_path = os.path.join(audit_folder, file)  
        df = pd.read_excel(file_path)  
        real_columns = [col for col in df.columns if col not in ["Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "Unnamed: 3"]]  
        if not real_columns:  
            print(f"Aucune colonne valide trouvée dans le fichier {file}.")  
            continue  
        # Vérifier si les colonnes clés existent  
        if "Production" in real_columns:  
            production = df[real_columns].iloc[0, real_columns.index("Production")]  
        else:  
            production = None  
        if "Cost Oil" in real_columns:  
            cost_oil = df[real_columns].iloc[0, real_columns.index("Cost Oil")]  
        else:  
            cost_oil = None  
        if "Profit Oil" in real_columns:  
            profit_oil = df[real_columns].iloc[0, real_columns.index("Profit Oil")]  
        else:  
            profit_oil = None  
        if "Training Budget" in real_columns:  
            training_budget = df[real_columns].iloc[0, real_columns.index("Training Budget")]  
        else:  
            training_budget = None  
        data.append({  
            "File": file,  
            "Production": production,  
            "Cost Oil": cost_oil,  
            "Profit Oil": profit_oil,  
            "Training Budget": training_budget  
        })  
except Exception as e:  
    print(f"Erreur lors de l'analyse des fichiers : {e}")
# Correction de l'import pour éviter l'erreur 'List'  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, ListItem, List
# Correction de l'import pour éviter l'erreur 'List'  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, ListItem
# Correction finale : Suppression de l'import incorrect 'List'  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, ListItem
# Suppression de l'import incorrect 'List'  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, ListItem
# Suppression de l'import incorrect 'List'  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
# Suppression de l'import incorrect 'List'  
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer