# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

Disciplina: Rețele Neuronale  
Instituție: POLITEHNICA București – FIIR  
Student: Laca Marian-George 
Link Repository GitHub: https://github.com/Marian-George-Laca24/Proiect-Retele-Neuronale.git
Data: 09/12/2025

---

1. Descrierea Proiectului

Acest proiect implementează un Sistem cu Inteligență Artificială (SIA) capabil să detecteze automat defectele de suprafață de pe sticlă industrială, clasificând imaginile în cinci clase distincte:

Scratch (zgârieturi)

Crack (fisuri)

Inclusion (impurități / incluziuni)

Bubble (bule de aer)

OK (sticlă fără defect)



2. Tabel Nevoie Reală → Soluție SIA → Modul Software

Tabelul următor sintetizează legătura directă dintre problema industrială reală și modulele software dezvoltate în proiect.

| **Nevoie reală concretă**                                       | **Cum o rezolvă SIA-ul**                                    | **Modul software responsabil**|
| --------------------------------------------------------------------- | ----------------------------------------------------------- |                               |  
| Detectarea automată a defectelor pe suprafețele de sticlă industrială | Clasificare imagine → tip defect detectat în < 1 secundă    | Modul RN + UI                 |
| Separarea automată a produselor conforme de cele defecte              | Sistemul clasifică imaginea în OK sau DEFECT și trimite     | RN + UI                       |
|                                                                       |                rezultatul către operator                    |                               |
| Eliminarea erorilor umane în procesul vizual                          | Modelul învață tiparele defectelor → consistență > 95%      | Data Logging + RN             |
|                                                                       |                 după antrenarea finală                      |                               |


3. Contribuția Originală la Setul de Date
Situația datasetului:

Total imagini finale (după preprocesare): ~578

Imagini originale capturate manual: ~443

Tipul contribuției:

 Date achiziționate cu senzori proprii telefon, set-up manual și imagini primite în cadrul practicii din vara anului 2025 (Saint-Gobain)
 Etichetare/adnotare manuală


Descriere detaliată:

Contribuția originală a reprezentat aproximativ 15–20% din dataset, prin captură proprie de imagini și etichetare manuală, restul datelor fiind preluate din dataseturi publice. În Etapa 5 se va extinde contribuția originală prin generare suplimentară de imagini.

O parte semnificativă a datasetului final a fost realizată manual prin fotografierea unor suprafețe reale de sticlă, sau prin intermediul firmei la care am facut practica in vara 2025(saint-gobain): ecran de telefon spart, sticlă zgâriată, panou din sticlă transparentă, zone cu incluziuni sau praf. Aceste imagini au fost capturate în condiții diferite de lumină, distanță și unghi, simulând diversitatea întâlnită în producția industrială.



4. Diagrama State Machine a Sistemului (Etapa 4)

Fișierul complet: docs/state_machine.png

### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **clasificare imagini cu input de la utilizator** deoarece aplicația mea are ca scop **detectarea automată a defectelor vizuale pe suprafațele din sticlă, indiferent de tipul de sticlă (telefon, geam, sticlă securizată etc.)**. Sistemul permite utilizatorului să încarce o imagine, să o proceseze automat printr-o rețea neuronală și să primească rezultatul clasificării în timp real.

Stările principale sunt:
1. **START** – Aplicația este lansată, se inițializează interfața și se verifică existența modelului de rețea neuronală.
2. **IDLE** – Sistemul așteaptă o acțiune din partea utilizatorului (încărcarea unei imagini).
3. **LOAD IMAGE** – Utilizatorul încarcă o imagine cu o suprafață de sticlă.
4. **PREPROCESS** – Imaginea este redimensionată, normalizată și pregătită pentru inferența rețelei neuronale.
5. **RN INFERENCE** – Modelul de rețea neuronală analizează imaginea și clasifică tipul de defect (ex: Scratch, Bubble, Crack, Inclusion, OK).
6. **DISPLAY RESULT** – Rezultatul este afișat utilizatorului împreună cu scorul de încredere (confidence).
7. **WAIT FOR NEXT MOVE** – Sistemul așteaptă o nouă imagine sau o nouă comandă din partea utilizatorului.

Tranzițiile critice sunt:
- **START → IDLE**: când aplicația pornește corect.
- **IDLE → LOAD IMAGE**: când utilizatorul selectează o imagine.
- **LOAD IMAGE → PREPROCESS**: când imaginea este validată corect.
- **PREPROCESS → RN INFERENCE**: când datele sunt gata pentru analiză.
- **RN INFERENCE → DISPLAY RESULT**: când clasificarea a fost realizată.
- **DISPLAY RESULT → WAIT FOR NEXT MOVE**: când utilizatorul a vizualizat rezultatul.
- **WAIT FOR NEXT MOVE → IDLE**: pentru inițierea unui nou ciclu de analiză.

Starea **ERROR** este esențială deoarece pot apărea erori precum:
- încărcarea unei imagini corupte,
- format de imagine incompatibil,
- lipsa modelului de rețea neuronală,
- erori la procesarea imaginilor.
În aceste situații, sistemul intră în starea ERROR, notifică utilizatorul și revine automat în starea IDLE pentru reluarea procesului în condiții sigure.

Bucla de feedback funcționează astfel: după afișarea rezultatului, utilizatorul poate încărca o nouă imagine, iar sistemul reia automat ciclul complet de procesare, permițând detectarea continuă a defectelor vizuale pe suprafețele din sticlă.



5. Modul 1: Data Logging / Acquisition

Locație: src/data_acquisition/

Funcționalități:

generează și structurează date în formatul necesar Etapei 3

include scripturile care au stat la baza datasetului (copiere, balansare, split)

integrează imaginile originale în structura finală

Cerințe îndeplinite:

✔ Codul rulează fără erori
✔ Format compatibil cu ImageDataGenerator
✔ Documentație minimală inclusă

6. Modul 2: Neural Network (Arhitectura RN)

Pentru rezolvarea problemei de clasificare a imaginilor cu defecte pe sticlă, a fost aleasă o arhitectură CNN modernă, bazată pe transfer learning.

Locație: src/neural_network/

Arhitectură:

Modelul este construit pe MobileNetV2, un backbone rapid și eficient pentru clasificarea imaginilor.

base_model = MobileNetV2(weights="imagenet", include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(5, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)


Stare în Etapa 4:

✔ Modelul este definit
✔ Modelul este compilat
✔ Modelul poate fi încărcat/salvat
❗ Antrenarea completă NU este finalizata


Pipeline-ul complet al aplicației este:

DATA → PREPROCESSING → NEURAL NETWORK → USER INTERFACE

Datele sunt încărcate din folderul data/, preprocesate în src/preprocessing/, analizate de modelul din src/neural_network/, iar rezultatul este afișat utilizatorului prin interfața din src/app/.


Structura Finală a Repository-ului (Etapa 4)

Proiect-Retele-Neuronale/
├── data/                     # Dataset utilizat în proiect
│   ├── classes/              # Imagini inițiale organizate pe clase de defect
│   └── split_balanced/       # Dataset final împărțit în train / validation / test
│
├── docs/                     # Documentație proiect
│   └── state_machine.png    # Diagrama State Machine a aplicației
│
├── src/                    
│   ├── data_acquisition/   
│   ├── preprocessing/       # Preprocesare + balansare dataset
│   ├── neural_network/      # Modul 2 – Rețea Neurală (train + predict)
│
├── models/                   # Modelele salvate
│   └── model.h5
│
├── README.md
├── README_Etapa4_Arhitectura_SIA.md
└── requirements.txt



7. Instrucțiuni de Instalare și Rulare (Etapa 4 – fără UI)
⚙️ Cerințe de Sistem

Sistem de operare: Windows 10 / Windows 11

Limbaj: Python 3.10

Mediu virtual recomandat: venv

Pasul 1: Activarea mediului virtual

Deschide PowerShell în folderul proiectului și rulează:

.\venv\Scripts\activate


Dacă mediul este activ, vei vedea (venv) în fața promptului.

Pasul 2: Instalarea dependențelor

Instalează toate bibliotecile necesare din fișierul requirements.txt:

py -3.10 -m pip install -r requirements.txt


⚠ Dacă unele pachete sunt deja instalate, pip va afișa mesaje de tip:
Requirement already satisfied — acest lucru este normal și corect.

Pasul 3: Verificare instalare biblioteci

Poți verifica pachetele instalate cu:

py -3.10 -m pip list


Trebuie să existe cel puțin:

tensorflow

keras

numpy

opencv-python

matplotlib

scikit-learn

Pasul 4: Rularea modulului de antrenare / testare RN

Pentru rularea codului de rețea neuronală:

py -3.10 src/neural_network/train_model.py


✔ Modelul va fi definit
✔ Modelul va fi compilat
✔ Modelul va fi salvat în folderul models/
✔ Se vor genera graficele:

training_accuracy.png

training_loss.png




