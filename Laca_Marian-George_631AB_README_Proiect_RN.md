## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | **Laca Marian-George** |
| **Grupa / Specializare** | **631 / Informatică Industrială** |
| **Disciplina** | **Rețele Neuronale** |
| **Instituție** | **Universitatea POLITEHNICA din București – FIIR** |
| **Link Repository GitHub** | https://github.com/Marian-George-Laca24/Proiect-Retele-Neuronale.git |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (PyTorch, Ultralytics YOLO, Roboflow, Streamlit) + CUDA (GPU acceleration) |
| **Domeniul Industrial de Interes (DII)** | Producție industrială – Controlul calității |
| **Tip Rețea Neuronală** | CNN – Object Detection (YOLO) |

---

## Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (evaluare prin mAP@50) | ≥70% | ~71% | **~75% (YOLO26s)** | +4% | ✓ |
| F1-Score (Macro) | ≥0.65 | ~0.69 | **~0.72** | +0.03 | ✓ |
| Latență inferență | <50 ms | ~14 ms | ~13 ms | −1 ms | ✓ |
| Contribuție date originale | ≥40% | **76.6%** | **76.6%** | – | ✓ |
| Nr. experimente optimizare | ≥4 | 4 | 5 | – | ✓ |

---

## Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, raționamentele și deciziile mele proprii.**

Asistenții de inteligență artificială (ex: ChatGPT) au fost utilizați **exclusiv ca instrumente de suport** pentru:
- clarificări teoretice,
- structurarea documentației,
- sugestii de organizare,
- debugging punctual.

**Nu au fost utilizate** soluții generate integral de AI pentru:
- arhitectura finală a rețelei neuronale,
- dataset,
- codul complet de antrenare sau evaluare,
- interpretarea rezultatelor.

---

### Confirmare explicită (bifez doar ce este adevărat):

| Nr. | Cerință | Confirmare |
|-----|---------|------------|
| 1 | Model RN antrenat de la zero (weights de la ultralytics) | [✓] DA |
| 2 | Minimum 40% date contribuție originală | [✓] DA |
| 3 | Cod propriu sau surse citate explicit | [✓] DA |
| 4 | Arhitectură și interpretare rezultate proprii | [✓] DA |
| 5 | Pot explica și justifica fiecare decizie tehnică | [✓] DA |

**Semnătură student:** *Laca Marian-George*

---

## 2. Descrierea Nevoii și Soluția SIA

## 2.1 Nevoia Reală / Studiul de Caz

În industria sticlei, controlul calității se realizează în mod tradițional prin **inspecție vizuală manuală**, proces care este consumator de timp, subiectiv și predispus la erori umane. Defecte precum **fisuri, zgârieturi, bule de aer sau incluziuni** pot fi dificil de identificat în condiții reale de producție, mai ales în prezența iluminării neuniforme sau a reflexiilor.

Scopul acestui proiect este dezvoltarea unei **Soluții Inteligente Asistate (SIA)** capabile să detecteze automat defectele de suprafață ale sticlei, folosind rețele neuronale convoluționale, reducând astfel dependența de factorul uman și crescând consistența procesului de inspecție.

---

## 2.2 Beneficii Măsurabile Urmărite

1. Reducerea timpului de inspecție manuală cu **peste 60%**
2. Detectarea defectelor cu **mAP@50 ≥70%**
3. Reducerea ratei de erori umane în controlul calității
4. Creșterea trasabilității prin logarea predicțiilor
5. Posibilitate de integrare ulterioară în fluxuri industriale reale

---

## 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Detectarea defectelor pe sticlă | Detecție obiecte în imagini RGB | YOLO (CNN) | mAP@50, Recall |
| Clasificare tip defect | Clasificare multi-clasă | YOLO Head | Precision per clasă |
| Identificare sticlă OK | Lipsa bounding box (background) | Post-procesare | FP rate |
| Viteză industrială | Inferență GPU optimizată | PyTorch + CUDA | <20 ms |

---

## 3. Dataset și Contribuție Originală

## 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt (dataset-uri publice + contribuție proprie) |
| **Surse** | Kaggle, Roboflow + imagini proprii |
| **Număr total imagini** | **578** |
| **Tip date** | Imagini RGB |
| **Format fișiere** | JPG / PNG + etichete YOLO |
| **Rezoluție utilizată** | 1024 × 1024 px |
| **Perioada colectării** | Decembrie 2025 – Ianuarie 2026 |

---

## 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 578 |
| **Observații originale (M)** | 443 |
| **Procent contribuție originală** | **76.6%** |
| **Tip contribuție** | Achiziție imagini + Poze proprii + etichetare manuală |
| **Instrument adnotare** | Roboflow |
| **Locație date** | `data/yolo26/` |

### Descriere metodă de achiziție și adnotare

Imaginile originale au fost colectate manual și includ atât suprafețe de sticlă **cu defecte reale** (fisuri, zgârieturi), cât și imagini **fără defecte (OK)**, utilizate pentru a învăța modelul să distingă între sticlă conformă și neconformă.

Toate cele **578 de imagini** au fost **adnotate manual** de catre autor, fiecare defect fiind încadrat prin bounding box și asociat clasei corespunzătoare, asigurând un control complet asupra calității etichetelor.

---

## 3.3 Preprocesare și Split Date

| Set | Procent | Număr imagini |
|-----|---------|---------------|
| Train | 70% | ~404 |
| Validation | 15% | ~87 |
| Test | 15% | ~87 |

**Preprocesări aplicate:**
- conversie la format RGB
- redimensionare la 1024 px
- normalizare automată specifică YOLO
- verificare consistență etichete
- split stratificat pe clase

---


## 4. Arhitectura SIA și State Machine

Această secțiune descrie arhitectura Sistemului cu Inteligență Artificială (SIA) dezvoltat în cadrul proiectului, precum și logica de funcționare a aplicației prin intermediul unei **State Machine**. Arhitectura este gândită pentru a simula un flux industrial real de inspecție vizuală a suprafețelor din sticlă.

---

## 4.1 Cele 3 Module Software

Aplicația este structurată modular, conform cerințelor disciplinei, în **trei module software principale**, fiecare având un rol clar definit în pipeline-ul general.

| Modul | Tehnologie | Funcționalitate principală | Locație în repository |
|------|------------|----------------------------|-----------------------|
| **Data Logging / Data Acquisition** | Python | Colectare date, organizare dataset, preprocesare, split train/val/test | `data/`, `src/preprocessing_tensorflow/` |
| **Neural Network (RN)** | TensorFlow (legacy) / PyTorch + YOLO | Antrenare, evaluare și inferență pentru detecția defectelor | `src/neural_network/`, `src/yolo8/`, `src/yolo26/` |
| **UI / Aplicație Software** | Streamlit | Interfață utilizator pentru încărcare imagini și afișare rezultate | `src/app/` |

**Observație:**  
În Etapele 4–5 este utilizată o rețea CNN implementată în TensorFlow, iar în Etapa 6 se face tranziția către modele YOLO (YOLOv8 și YOLO26), pentru îmbunătățirea performanței și realismului industrial.

---

## 4.2 State Machine – Descriere și Flux Logic

### Locație diagramă
- `docs/state_machine.png`

Diagrama State Machine reprezintă **logica de control a aplicației**, de la pornire până la afișarea rezultatului și reluarea ciclului de analiză.




### Stările definite în aplicație

State Machine-ul implementat conține următoarele stări:

| Stare | Descriere |
|------|-----------|
| **START** | Inițializarea aplicației și a componentelor interne |
| **IDLE** | Aplicația așteaptă o acțiune din partea utilizatorului |
| **LOAD IMAGE** | Încărcarea unei imagini cu suprafață de sticlă |
| **PREPROCESS** | Redimensionare, normalizare și pregătire imagine |
| **RN INFERENCE** | Inferență folosind rețeaua neuronală |
| **DISPLAY RESULT** | Afișarea rezultatului (clasă + confidence) |
| **WAIT FOR NEXT MOVE** | Așteptare pentru o nouă analiză |
| **ERROR** | Gestionarea erorilor apărute în pipeline |

---

### Tranziții principale între stări

- **START → IDLE**  
  Aplicația pornește, iar modelul de rețea neuronală este încărcat în memorie.

- **IDLE → LOAD IMAGE**  
  Utilizatorul selectează o imagine pentru analiză.

- **LOAD IMAGE → PREPROCESS**  
  Imaginea este validată și pregătită pentru inferență.

- **PREPROCESS → RN INFERENCE**  
  Imaginea este transmisă rețelei neuronale pentru analiză.

- **RN INFERENCE → DISPLAY RESULT**  
  Modelul returnează predicția, care este afișată utilizatorului.

- **DISPLAY RESULT → WAIT FOR NEXT MOVE**  
  Utilizatorul vizualizează rezultatul și poate continua.

- **WAIT FOR NEXT MOVE → IDLE**  
  Sistemul revine în stare de așteptare pentru o nouă imagine.

- **ORICE STARE → ERROR**  
  Dacă apare o eroare (imagine invalidă, lipsă model, incompatibilitate), sistemul intră în starea ERROR.

- **ERROR → IDLE**  
  După notificarea utilizatorului, aplicația revine în stare sigură.

---

## 4.3 Justificarea Alegerii State Machine-ului

Alegerea unei arhitecturi bazate pe **State Machine** este justificată de natura aplicației și de cerințele unui flux industrial de inspecție vizuală:

- control clar și predictibil al fluxului de execuție,
- separarea logică a etapelor (input, procesare, inferență, output),
- gestionarea robustă a erorilor,
- posibilitatea extinderii ulterioare (ex. logging industrial, integrare PLC, decizii automate).

Această structură permite rularea repetitivă și sigură a procesului de detecție a defectelor, simulând modul de operare al unui sistem real de control al calității din industrie.

---


## 5. Modelul RN – Antrenare și Optimizare

Această secțiune descrie arhitectura rețelei neuronale utilizate, procesul de antrenare, precum și experimentele de optimizare realizate în cadrul proiectului. Evoluția modelului reflectă o tranziție progresivă de la o soluție de bază către una optimizată, cu performanțe superioare și relevanță industrială crescută.

---

## 5.1 Arhitectura Rețelei Neuronale

### Notă privind experimente preliminare (TensorFlow)

În fazele inițiale ale proiectului a fost explorată și o abordare bazată pe o rețea neuronală convoluțională implementată în **TensorFlow/Keras**, utilizând un model de clasificare a imaginilor.

Această variantă a fost utilizată exclusiv **în scop exploratoriu**, pentru înțelegerea comportamentului datelor și a dificultății problemei (defecte vizuale mici, cu variații mari de formă și textură).

În urma acestor teste preliminare, s-a concluzionat că problema abordată este una de **object detection**, nu de simplă clasificare, motiv pentru care soluția finală a fost reproiectată folosind arhitectura YOLO, mult mai potrivită pentru localizarea și clasificarea simultană a defectelor.


### Evoluția arhitecturii utilizate

În cadrul proiectului au fost utilizate **două generații de modele YOLO**, după cum urmează:

1. **YOLOv8n** – model inițial (baseline experimental)
2. **YOLO26 (YOLO26n și YOLO26s)** – model final și optimizat

Această tranziție a fost motivată de necesitatea obținerii unor:
- valori mai bune pentru mAP@50–95,
- performanțe superioare pe clase dificil de detectat (ex: *inclusion*),
- rezultate mai stabile în context industrial.

---

### Arhitectura YOLO (conceptual)

Input Image (1024 × 1024)
↓
Backbone CNN (extracție caracteristici multi-scale)
↓
Neck (Feature Pyramid Network)
↓
Detection Head
├── Bounding Box Regression
├── Objectness Score
└── Clasificare multi-clasă
Output: coordonate + clasă + confidence


Modelul este de tip **CNN end-to-end pentru object detection**, capabil să identifice simultan:
- locația defectului (bounding box),
- tipul defectului (bubble, crack, inclusion, scratch),
- scorul de încredere (confidence).

---

### Justificarea alegerii YOLO

YOLO a fost ales deoarece:
- permite **detecție în timp aproape real**,
- este standard industrial pentru inspecție vizuală,
- oferă un compromis optim între **acuratețe, viteză și complexitate**,
- suportă fine-tuning și extensii moderne (YOLO26).

---

## 5.2 Configurația de Antrenare (Model Optimizat – Etapa 6)

### Parametri principali utilizați

| Parametru | Valoare Finală | Justificare |
|----------|----------------|-------------|
| Image Size (`imgsz`) | 1024 | Detalii fine ale defectelor pe sticlă |
| Batch Size | 8 (YOLO26s) | Maxim stabil pe GPU RTX 4060 (8GB VRAM) |
| Epochs | max 120 | Control prin Early Stopping |
| Optimizer | AdamW (implicit YOLO) | Convergență stabilă |
| Loss | YOLO Detection Loss | Detecție multi-clasă |
| Early Stopping | activ (patience=20) | Prevenire overfitting |
| Device | GPU (CUDA) | Accelerare antrenare |

---

### Platformă hardware utilizată

- **GPU:** NVIDIA RTX 4060 Laptop (8 GB GDDR6)
- **RAM:** 32 GB
- **Framework:** Ultralytics YOLO (PyTorch)
- **OS:** Windows

---

## 5.3 Experimente de Optimizare

În cadrul Etapei 6 au fost realizate **minimum 4 experimente distincte**, conform cerințelor.

### Tabel sinteză experimente

| Exp # | Model | Configurație | mAP@50 | mAP@50–95 | Observații |
|-----|------|--------------|--------|-----------|------------|
| Baseline | YOLOv8n | imgsz=1024, batch=8 | ~0.69 | ~0.48 | Performanță limitată |
| Exp 1 | YOLO26n | batch=-1 (auto) | 0.71 | 0.53 | Stabil, dar limitat pe clase dificile |
| Exp 2 | YOLO26n + fine-tune | Early stopping | 0.71 | 0.52 | Fără câștig suplimentar |
| Exp 3 | YOLO26s | batch=8 | **0.75** | **0.53** | Creștere clară performanță |
| Exp 4 | YOLO26s + fine-tune | Early stopping | 0.72 | 0.49 | Overfitting ușor |

---

### Observații cheie din experimente

- **YOLO26n** oferă performanță bună, dar este limitat de capacitate.
- **YOLO26s** aduce îmbunătățiri clare pentru clasele *bubble* și *scratch*.
- Fine-tuning-ul agresiv nu a adus câștig suplimentar → dataset relativ mic.
- Early stopping a prevenit degradarea performanței pe setul de validare.

---

## 5.4 Alegerea Modelului Final

### Model selectat pentru producție:

**Modelul final utilizat în evaluarea proiectului este **YOLO26s (v5)**, antrenat de la varianta publica de la Ultralytics(yolo26s.pt) și ulterior optimizat prin fine-tuning cu rată de învățare redusă și mecanism de early stopping.**

**Motivele alegerii:**
- cel mai bun compromis între mAP și stabilitate,
- rezultate consistente pe toate clasele,
- latență acceptabilă pentru aplicație industrială,
- comportament robust în fața dezechilibrului de date.

---

### Referințe fișiere relevante

- `runs/detect/results_yolo/train_yolo26_v3_yolo26s/`
- `results/yolo26_best_metrics.json`
- `results/yolo26s_best_metrics.json`
- `docs/confusion_matrix_optimized.png`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

**Modelul final utilizat în evaluarea proiectului este **YOLO26s (v5)**, antrenat de la varianta publica de la Ultralytics(yolo26s.pt) și ulterior optimizat prin fine-tuning cu rată de învățare redusă și mecanism de early stopping.**

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy (≈ mAP@50)** | **74.4%** | ≥70% | ✓ |
| **F1-Score (Macro)** | **0.72** | ≥0.65 | ✓ |
| **Precision (Macro)** | **0.71** | - | - |
| **Recall (Macro)** | **0.73** | - | - |

**Notă:** Pentru modele de tip YOLO (object detection), Accuracy este echivalată cu **mAP@50**, metric standard în evaluarea performanței de detecție.

---

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline – YOLO26s v3) | Etapa 6 (Optimizat – v5) | Îmbunătățire |
|--------|-------------------------------|---------------------------|--------------|
| Accuracy (mAP@50) | ~72.2% | **74.4%** | **+2.2%** |
| F1-Score (Macro) | ~0.62 | **0.72** | **+0.10** |

**Referință fișier:** `results/yolo26s_best_metrics.json`


---

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|------|------------|
| **Clasa cu cea mai bună performanță** | `bubble` – Precision ≈ 82%, Recall ≈ 96% |
| **A doua clasă performantă** | `crack` – Precision ≈ 82%, Recall ≈ 87% |
| **Clasa cu cea mai slabă performanță** | `inclusion` – Recall ≈ 37% |
| **Confuzii frecvente** | `inclusion` confundată cu `background` |
| **Dezechilibru clase** | `inclusion` are număr redus de instanțe → impact negativ asupra recall-ului |

---

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | Incluziune mică, contrast scăzut | background | inclusion | Diferență vizuală minimă | Defect nedetectat |
| 2 | Defect fin pe suprafață texturată | scratch | inclusion | Similaritate texturală | Clasificare eronată |
| 3 | Imagine cu iluminare neuniformă | scratch | background | Zgomot de iluminare | Fals pozitiv |
| 4 | Defect parțial vizibil | background | scratch | Bounding box incomplet | Reinspecție manuală |
| 5 | Obiect marginal în cadru | scratch | crack | Context vizual insuficient | Alertă incorectă |

---

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

Într-un scenariu industrial de inspecție vizuală automată, modelul detectează corect aproximativ **73 din 100 de defecte reale** (Recall ≈ 73%). Defectele critice (`bubble`, `crack`) sunt identificate cu rate de peste **85%**, ceea ce face modelul adecvat ca sistem de suport decizional pentru operatori umani (*human-in-the-loop*).

Exemplu de impact:
- Din 100 piese defecte reale, ~27 pot necesita reinspecție manuală.
- Din 100 piese bune, aproximativ 8–10 pot fi marcate eronat ca defecte, generând costuri minore de reinspecție, dar prevenind livrarea produselor neconforme.

**Pragul de acceptabilitate pentru domeniu:**  
Recall ≥ 85% pentru defecte critice (`bubble`, `crack`)

**Status:** **Atins pentru defecte critice**  
**Plan de îmbunătățire (pentru clasa `inclusion`):**
- augmentare direcționată a datelor,
- colectare suplimentară de exemple reale,
- ajustare prag de decizie pentru clase cu risc scăzut.

---



## 7. Aplicația Software Finală

Aplicația software reprezintă componenta de interfață și integrare a Sistemului cu Inteligență Artificială (SIA), având rolul de a permite utilizatorului final să utilizeze modelul de detecție într-un mod intuitiv, reproductibil și sigur din punct de vedere operațional.

Aceasta este concepută ca o aplicație **end-to-end**, de la încărcarea imaginii până la afișarea rezultatului inferenței, utilizând **modelul YOLO26s optimizat**, antrenat special pentru detectarea defectelor în sticlă.

---

### 7.1 Modificări Implementate în Etapa 6

Tabelul de mai jos sintetizează modificările esențiale aduse aplicației software în etapa de optimizare (Etapa 6), comparativ cu versiunea din Etapa 5.

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model utilizat** | YOLOv8n | **YOLO26s (v5 - fine-tuned)** | Creștere semnificativă a mAP și stabilitate mai bună |
| **Model încărcat în UI** | `best.pt` (YOLOv8) | `best.pt` (YOLO26s fine-tuned) | Performanță superioară pe defecte reale |
| **Dimensiune input** | 640 px | **1024 px** | Detectarea defectelor mici |
| **Pipeline inferență** | Simplificat | **Validare + inferență + overlay rezultate** | Robustete crescută |
| **Afișare rezultat** | Etichetă text | **Bounding boxes + confidence** | Interpretabilitate pentru operator |
| **Logging** | Minimal | **Salvare rezultate și imagini** | Trasabilitate și audit |
| **Stare ERROR** | Parțial | **Gestionare completă** | Evitare crash aplicație |

---

### 7.2 Screenshot UI cu Model Optimizat

**Locație fișiere:**
- `docs/screenshots/inference_optimized1.png`
- `docs/screenshots/inference_optimized2.png`

**Descriere:**

Screenshot-urile prezintă interfața aplicației în timpul rulării inferenței pe imagini reale de sticlă, diferite de cele din seturile de train/validation/test.

Elemente vizibile:
- imaginea încărcată de utilizator,
- bounding box-uri desenate peste defecte,
- eticheta defectului detectat (ex: *crack*, *bubble*, *scratch*),
- scorul de încredere (confidence),
- comportament stabil indiferent de tipul imaginii.

Aceste capturi demonstrează faptul că aplicația folosește **modelul optimizat final**, nu o versiune intermediară.

---

### 7.3 Demonstrație Funcțională End-to-End

Aplicația a fost testată complet pe un flux end-to-end, utilizând imagini reale de sticlă neincluse în dataset-ul de antrenare.

**Locație dovadă (opțional):**
- `docs/demo/` (GIF / video / secvență screenshots)

#### Flux demonstrat

| Pas | Acțiune | Rezultat vizibil |
|----|--------|------------------|
| 1 | Încărcare imagine | Imagine afișată în UI |
| 2 | Validare input | Verificare format și dimensiune |
| 3 | Preprocesare | Redimensionare la 1024 px |
| 4 | Inferență YOLO | Detectare defecte |
| 5 | Afișare rezultat | Bounding boxes + confidence |
| 6 | Reset / nou input | Aplicația rămâne activă |

**Latență end-to-end măsurată:**  
~120–180 ms / imagine (RTX 4060, batch = 1)

**Data demonstrației:**  
11/02/2026

---

### 7.4 Robustete și Siguranță în Utilizare

Aplicația gestionează explicit următoarele situații:
- imagine invalidă sau coruptă,
- format neacceptat,
- lipsa modelului optimizat,
- eroare la inferență GPU.

În toate cazurile, sistemul:
- intră în starea `ERROR`,
- afișează un mesaj clar pentru utilizator,
- revine automat în starea `IDLE`, fără blocarea aplicației.

Această abordare este esențială pentru utilizare într-un context industrial real.

---


## 8. Structura Repository-ului Final

Structura finală a repository-ului a fost organizată astfel încât să reflecte:
- evoluția proiectului pe etape,
- separarea clară între implementările **legacy (TensorFlow)** și soluția **finală (YOLO26)**,
- trasabilitatea completă a experimentelor și rezultatelor,
- cerințele explicite pentru evaluarea finală la disciplina *Rețele Neuronale*.

---


Proiect-Retele-Neuronale-Marian-George-Laca24/
│
├── Laca_Marian-George_631AB_README_Proiect_RN.md        # Documentație FINALĂ – livrabil examen
│
├── docs/
│   ├── README – Etapa 3 -Analiza si Pregatirea Setului de Date pentru Retele Neuronale.md  # Etapa 3 – analiză și pregătire date
│   ├── README_Etapa4_Arhitectura_SIA.md                                # Etapa 4 – arhitectură + state machine
│   ├── README_Etapa5_Antrenare_RN.md                                   # Etapa 5 – antrenare baseline
│   ├── README_Etape6_Analiza_Performantei_Optimizare_Concluzii.md      # Etapa 6 – optimizare și concluzii
│   │
│   ├── demo.rar                            #arhiva cu videoclipul scurt demo
│   │
│   ├── state_machine.png                   # State Machine versiune finală
│   │
│   ├── confusion_matrix_optimized.png      # Confusion matrix – model FINAL
│   │
│   ├── screenshots/
│   │   ├── inference_optimized1.png        # UI + YOLO26s – inferență reală
│   │   └── inference_optimized2.png
│   │
│   ├── optimization/
│   │   ├── accuracy_comparison.png
│   │   ├── f1_comparison.png
│   │   └── learning_curves_best.png
│   │
│   └── results/
│       ├── learning_curves_final.png
│       ├── metrics_evolution.png
│       └── yolo_pred_IMG_*.png
│
├── data/
│   ├── raw/                                # Imagini brute (imagini proprii+descarcate din dataseturi publice)
│   │
│   ├── yolo26/                             # Dataset FINAL YOLO26
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── data.yaml
│   │
│   ├── yolo8/                             # Dataset Etapa 6 YOLO8
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── data.yaml
│
├── src/
│   ├── legacy_tensorflow/                  # IMPLEMENTARE a TensorFlow (Etapa 3–5)
│   │   ├── train_model.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── legacy_app_tf.py
│   │
│   ├── preprocessing_tensorflow/           # Scripturi inițiale pre-Roboflow si pre-yolo
│   │   ├── balance_dataset.py
│   │   ├── combine_datasets.py
│   │   ├── split_balanced_dataset.py
│   │   └── export_classes.py
│   │
│   ├── yolo26/                             # SOLUȚIA FINALĂ
│   │   ├── train_yolo26_v1.py               # YOLO26n – baseline
│   │   ├── train_yolo26_v2.py      # YOLO26n – fine-tune
│   │   ├── train_yolo26_v3_yolo26s.py       # YOLO26s – baseline
│   │   ├── train_yolo26_v4_finetune_yolo26s.py      # YOLO26s – fine-tune
│   │   ├── train_yolo26_v5_yolo26s_refined.py
│   │   ├── evaluate_yolo26.py
│   │   └── predict_yolo26.py
│   │
│   ├── yolo8/                               
│   │   ├── evaluate_yolo.py
│   │   ├── first_train_yolo.py
│   │   ├── plot_learning_curves.py
│   │   ├── plot_metrics_evolution.py
│   │   ├── plot_optimization.py
│   │   ├── predict_yolo.py
│   │   └── train_yolo.py
│   │
│   ├── app/
│   │   ├── legacy_app_tf.py
│   │   └── app_yolo.py                     # Aplicația FINALĂ (YOLO26)
│
├── models/
│   ├── yolo26/
│   │   ├── best.pt                         # MODEL FINAL FOLOSIT ÎN UI
│   │   └── last.pt                       
│   │
│   ├── yolo8/
│   │   ├── best.pt
│   │   └── last.pt 
│   │
│   └── legacy_tensorflow/
│       ├── model.h5
│       └── trained_model.h5
│
├── results/
│   ├── training_experiments.csv            # Tabel experimente YOLO - training yolo26s
│   ├── optimization_experiments.csv        # Tabel experimente YOLO - optimizare yolo26s
│   ├── yolo_test_metrics_v13.json
│   ├── yolo_test_metrics_v14.json
│   ├── yolo26_best_metrics.json
│   ├── yolo26s_best_metrics.json            # Metrici finale YOLO26s
│   └── error_analysis.json
│
├── results_yolo/                            # Output Ultralytics, generat la fiecare experiment Etapa 5 si etapa 6 
│   ├── train_v1/                            
│   ├── train_v12/
│   ├── train_v13/
│   ├── train_v13_2/
│   ├── train_v14/
│   ├── train_yolo26_v1/
│   ├── train_yolo26_v2_finetune_best/
│   ├── train_yolo26_v3_yolo26s/
│   ├── train_yolo26_v4_finetune_yolo26s/
│   └── train_yolo26_v5_finetune_yolo26s/
│
├── requirements.txt
└── .gitignore


---


## 9. Instrucțiuni de Instalare și Rulare

Această secțiune descrie pașii necesari pentru instalarea mediului de lucru,
reproducerea experimentelor principale și rularea aplicației finale bazate pe
modelul **YOLO26s optimizat** (Etapa 6).

---

### 9.1 Cerințe Preliminare

Pentru rularea proiectului sunt necesare următoarele condiții minime:

- **Python ≥ 3.8** (recomandat Python 3.10+)
- **pip ≥ 21.0**
- Sistem de operare: **Windows / Linux / macOS**
- **GPU NVIDIA compatibil CUDA** (recomandat, dar nu obligatoriu)

**Notă:**  
Proiectul **NU necesită LabVIEW**. Toate componentele sunt implementate exclusiv
în limbajul **Python**, utilizând framework-uri moderne de deep learning.
Framework principal: PyTorch + Ultralytics YOLO
TensorFlow este păstrat doar pentru reproducerea Etapei 5.

---

### 9.2 Instalare

Pentru instalarea proiectului se urmează pașii de mai jos:

```bash

# 1. Clonare repository
git clone https://github.com/Marian-George-Laca24/Proiect-Retele-Neuronale
cd Proiect-Retele-Neuronale-Marian-George-Laca24

# 2. Creare mediu virtual (recomandat)
python -m venv venv
venv\Scripts\activate        # Windows
# sau:
source venv/bin/activate    # Linux / macOS

# 3. Instalare dependențe
pip install -r requirements.txt

```

Toate bibliotecile necesare (Ultralytics YOLO, PyTorch, NumPy, OpenCV etc.)
sunt definite explicit în fișierul requirements.txt.


### 9.3 Rulare Pipeline Complet (YOLO26 – Soluția Finală)

Această secțiune descrie rularea completă a pipeline-ului, de la verificarea
datasetului până la inferența finală cu modelul optimizat.

#### Pasul 1: Verificare structură dataset

Datasetul final este deja pregătit și structurat conform cerințelor YOLO în: data/yolo26/

Structura include:

train/ – set de antrenare

valid/ – set de validare

test/ – set de test

data.yaml – definirea claselor și a path-urilor

Nu este necesară rularea niciunui script suplimentar de preprocesare.


#### Pasul 2: Antrenare YOLO26s – Baseline (opțional, pentru reproducere)

Pentru reproducerea experimentului de bază (YOLO26s fără fine-tuning):
    python src/yolo26/train_yolo26_v3_yolo26s.py

Rezultatele sunt salvate automat în:
    results_yolo/train_yolo26_v3_yolo26s/



#### Pasul 3: Fine-tuning YOLO26s – Model Optimizat (Etapa 6)

Pentru rularea etapei finale de optimizare:

python src/yolo26/train_yolo26_v4_finetune_yolo26s.py


Rezultatele sunt salvate în:

results_yolo/train_yolo26_v4_finetune_yolo26s/


Modelul final optimizat (best.pt) este utilizat în aplicația software.


#### Pasul 4: Evaluare model final pe test set
python src/yolo26/evaluate_yolo26.py


Metricile obținute sunt salvate în:

results/yolo26s_best_metrics.json


#### Pasul 5: Rulare aplicație finală (inferință)
streamlit run src/app/app_yolo.py


Aplicația încarcă automat modelul final:

models/yolo26/best.pt


și permite inferența pe imagini noi, neutilizate în antrenare.


### 9.4 Verificare Rapidă (Smoke Test)
# Verificare încărcare model YOLO26s
python -c "from ultralytics import YOLO; YOLO('models/yolo26/best.pt'); print('✓ Model YOLO26s încărcat cu succes')"


Această comandă confirmă funcționarea corectă a mediului și a modelului final.

### 9.5 Observații Importante

Folderul results_yolo/ conține toate experimentele YOLO și este generat automat

Folderul legacy_tensorflow/ reprezintă implementări istorice (Etapa 3–5)

Aplicația finală folosește exclusiv YOLO26s optimizat

Toate path-urile sunt relative, compatibile cu rularea pe orice sistem

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| Automatizarea detecției defectelor de suprafață | Funcțional | Da | ✓ |
| Detectarea defectelor critice (crack, scratch) | Recall ≥ 70% | Da | ✓ |
| Accuracy pe test set | ≥ 70% | ~74% | ✓ |
| F1-Score (Macro) | ≥ 0.65 | ~0.71 | ✓ |
| Timp inferență sub 20 ms / imagine | ≤ 20 ms | ~12 ms | ✓ |

Rezultatele finale confirmă atingerea tuturor obiectivelor inițiale stabilite
în Secțiunea 2. Modelul YOLO26s optimizat oferă un compromis foarte bun între
acuratețe, viteză de inferență și robustețe în condiții reale.

---

### 10.2 Ce NU Funcționează – Limitări Cunoscute

Evaluatorul apreciază identificarea clară a limitărilor, acestea fiind prezentate
transparent mai jos:

1. **Iluminare slabă sau neuniformă**  
   Performanța scade în cazul imaginilor cu iluminare foarte redusă sau reflexii
   puternice, în special pentru clasa *inclusion*.

2. **Dezechilibru de clase**  
   Clasa *inclusion* are un număr mai redus de exemple, ceea ce conduce la un recall
   mai mic comparativ cu *crack* și *scratch*.

3. **False Positive pentru zgârieturi fine**  
   În anumite cazuri, texturi naturale ale suprafeței sunt confundate cu defecte
   de tip *scratch*.

4. **Funcționalități neimplementate**  
   Export ONNX și integrare API REST pentru deployment industrial nu au fost
   implementate din lipsă de timp.

---

### 10.3 Lecții Învățate (Top 5)

1. **Importanța calității datasetului**  
   Calitatea și diversitatea imaginilor au avut un impact mai mare decât creșterea
   numărului de epoci.

2. **Early Stopping este esențial**  
   Oprirea timpurie a prevenit overfitting-ul și a redus timpul total de antrenare.

3. **Fine-tuning > antrenare de la zero pentru modele YOLO**  
   Fine-tuning-ul YOLO26s a adus câștiguri clare față de varianta YOLO26n.

4. **Augmentările generice nu sunt suficiente**  
   Augmentările specifice domeniului industrial sunt esențiale pentru generalizare.

5. **Documentarea incrementală economisește timp**  
   Structurarea proiectului pe etape a facilitat integrarea finală și redactarea
   documentației pentru examen.

---

### 10.4 Retrospectivă

Dacă proiectul ar fi reluat, una dintre primele modificări ar fi colectarea unui
număr mai mare de imagini originale pentru clasa *inclusion*, încă din etapele
inițiale. De asemenea, ar fi fost utilă definirea mai timpurie a unui protocol
standardizat de captură a imaginilor (iluminare, unghi, distanță).

În plus, integrarea timpurie a unui mecanism de evaluare continuă pe un set fix
de imagini reale ar fi permis identificarea mai rapidă a limitărilor modelului.

---

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1–2 săptămâni) | Augmentare date pentru clasa *inclusion* | +10–15% recall |
| **Medium-term** (1–2 luni) | Ensemble YOLO26s + YOLO26m | +3–5% accuracy |
| **Long-term** | Deployment pe edge device (Jetson / RPi) | Inferență <15 ms |

---

## 11. Bibliografie

1. **Abaza, Bogdan Felician**, *Curs Rețele Neuronale – Facultatea de Inginerie Industrială și Robotică*,  
   POLITEHNICA București, 2024–2025. Material didactic intern.

2. Ralph Christian Detchosa, Glass Dataset Bubbles Computer Vision Dataset, 2025 .
   https://universe.roboflow.com/ralph-christian-detchosa/glass-dataset-bubbles-j2bmg

3. labelled good, Inclusion Computer Vision Dataset , 2021 .
   https://universe.roboflow.com/ralph-christian-detchosa/glass-dataset-bubbles-j2bmg

4. Ultralytics, **Ultralytics YOLO26**, 2026.  
   https://docs.ultralytics.com/models/yolo26/


---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set
- [x] **F1-Score ≥0.65** pe test set
- [x] **Contribuție ≥40% date originale**
- [x] **Fine-tuning controlat pe weights publice(Ultralytics)**
- [x] **Minimum 4 experimente de optimizare**
- [x] **Confusion matrix generată și interpretată**
- [x] **State Machine definit**
- [x] **Cele 3 module funcționale (Date, RN, UI)**
- [x] **Demonstrație end-to-end funcțională**

### Repository și Documentație

- [x] **README.md complet**
- [x] **README-uri pentru Etapele 3–6**
- [x] **Screenshots și rezultate incluse**
- [x] **Structură repository conformă**
- [x] **requirements.txt actualizat**
- [x] **Path-uri relative**

### Acces și Versionare

- [x] **Repository accesibil cadrelor didactice**
- [x] **Tag `v0.6-optimized-final`**
- [x] **Istoric commit-uri incremental**

---

## Note Finale

**Versiune document:** FINAL – Examen Rețele Neuronale  
**Ultima actualizare:** 02.02.2026  
**Tag Git:** `v0.6-optimized-final`

---

*Acest README reprezintă documentația principală pentru Livrabilul 1 – Aplicație
Software bazată pe Rețele Neuronale. Pentru Livrabilul 2 (prezentare PowerPoint),
consultați structura indicată în documentul RN_Specificatii_proiect.pdf.*
