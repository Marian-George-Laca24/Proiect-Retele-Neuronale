# README – Etapa 3 -Analiza si Pregatirea Setului de Date pentru Retele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Laca Marian-George
**Link Repository GitHub:** https://github.com/Marian-George-Laca24/Proiect-Retele-Neuronale.git 
**Data predării:** 15.01.2026

---
---

## Introducere

Această etapă descrie procesul de colectare, analiză și pregătire a setului de date utilizat în proiectul de detecție a defectelor în sticlă. Scopul Etapei 3 este obținerea unui dataset coerent, bine documentat și compatibil cu antrenarea rețelelor neuronale utilizate ulterior în proiect.

Față de etapele inițiale ale proiectului, această versiune a dataset-ului reflectă o schimbare de abordare, prin trecerea de la un set de date preluat din surse publice la un dataset majoritar original, colectat și adnotat manual, în scop didactic.

---

## 1. Structura Repository-ului (Etapa 3)

proiect-rn-[Marian-George-Laca24]/
├── data/
│ ├── raw/ # imagini brute, neprocesate
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ ├── valid/
│ │ ├── images/
│ │ └── labels/
│ ├── test/
│ │ ├── images/
│ │ └── labels/
│ └── data.yaml # definirea claselor și a path-urilor (YOLO)
│
├── src/
│ └── preprocessing/ # scripturi inițiale de preprocesare (istoric)
│
├── docs/
│ └── datasets/ # documentație dataset (opțional)
│
└── requirements.txt



---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

- **Origine:** Dataset propriu, colectat de student  
- **Mod de achiziție:** Fotografiere directă a probelor de sticlă  
- **Echipament:** Cameră foto telefon mobil  
- **Mediu:** Non-industrial (condiții de laborator / casnice)  
- **Iluminare:** Nespecializată, variabilă  
- **Adnotare:** Manuală, realizată în platforma Roboflow  

Dataset-ul a fost construit în scop didactic, pentru a permite înțelegerea completă a fluxului de lucru dintr-un proiect de detecție vizuală.

---

### 2.2 Caracteristicile dataset-ului

- **Număr total imagini:** 578  
- **Număr imagini originale:** 443  
  - reprezintă aproximativ **76,6%** din dataset  
- **Tip date:** Imagini RGB  
- **Format fișiere:** JPG / PNG  
- **Rezoluție imagini:** variabilă  
- **Rezoluție de intrare în rețea:** 1024 px (redimensionare aplicată la antrenare)

---

### 2.3 Clasele dataset-ului

Dataset-ul conține **4 clase de defecte**, specifice industriei sticlei:

- `bubble` – bule de aer  
- `crack` – fisuri  
- `scratch` – zgârieturi  
- `inclusion` – incluziuni / impurități  

În plus, dataset-ul include și **imagini cu sticlă intactă (OK)**, care:

- nu reprezintă o clasă de defect,
- sunt utilizate ca exemple negative,
- sunt importante pentru reducerea detecțiilor false (false positives).

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Observații generale

Analiza exploratorie a dataset-ului a evidențiat:

- variații semnificative de iluminare între imagini,
- diferențe de contrast și textură ale suprafeței de sticlă,
- defecte de dimensiuni variabile, unele foarte mici,
- distribuție neuniformă a claselor.

### 3.2 Probleme identificate

- dezechilibru de clasă, în special pentru defectele de tip `inclusion`,
- defecte de dimensiuni reduse, dificil de distins de fundal,
- reflexii și zgomot vizual care pot fi confundate cu defecte reale,
- lipsa unui mediu industrial controlat.

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

Au fost aplicate următoarele operații:

- eliminarea imaginilor duplicate,
- verificarea și corectarea etichetelor greșite,
- uniformizarea denumirilor claselor,
- verificarea consistenței fișierelor de adnotare.

---

### 4.2 Adnotarea datelor

Adnotarea a fost realizată manual, pentru toate cele **578 de imagini**, folosind platforma Roboflow:

- fiecare defect a fost încadrat cu bounding box,
- fiecare bounding box a fost asociat unei clase,
- procesul a fost realizat integral de student.

Acest tip de adnotare este fezabil în scop didactic, dar nu ar fi scalabil într-un flux industrial real fără instrumente semi-automate sau automate.

---

### 4.3 Structurarea dataset-ului

Dataset-ul a fost împărțit în:

- **Train:** ~70%  
- **Validation:** ~15%  
- **Test:** ~15%  

Principii respectate:

- separare clară între seturi,
- evitarea scurgerii de informație (data leakage),
- păstrarea proporțiilor relative între clase.

---

### 4.4 Format final

Datele sunt organizate conform standardului YOLO, utilizând:

- foldere separate pentru imagini și etichete,
- fișierul `data.yaml` care definește:
  - clasele,
  - path-urile pentru train / validation / test.

---

## 5. Fișiere Generate în Această Etapă

- `data/raw/` – imagini brute  
- `data/train/`, `data/valid/`, `data/test/` – seturi finale  
- `data.yaml` – configurare dataset YOLO  
- `src/preprocessing/` – scripturi inițiale de preprocesare (istoric)  

---

## 6. Stare Etapă

- [x] Structură repository configurată  
- [x] Dataset colectat și analizat  
- [x] Adnotare manuală completă  
- [x] Seturi train / validation / test generate  
- [x] Dataset pregătit pentru antrenarea modelelor de detecție  

---
