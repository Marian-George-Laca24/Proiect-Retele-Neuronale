Crearea și Pregătirea Setului de Date (Dataset)

Pentru proiectul „Detectarea Defectelor în Sticlă folosind Rețele Neuronale”, setul de date a fost construit prin combinarea mai multor dataset-uri publice care conțin imagini cu defecte specifice industriei sticlei. Dataset-ul final este format exclusiv din imagini (RGB) etichetate, împărțite ulterior în train, validation și test.

3.1. Colectarea dataset-urilor
Pentru a acoperi toate tipurile de defecte relevante, am selectat cinci categorii principale:
  
  -Scratches (zgârieturi)
  
  -Cracks (fisuri)
  
  -Inclusions (incluziuni/impurități în sticlă)
  
  -Bubbles (bule de aer)
  

Pentru fiecare categorie, au fost descărcate unul sau mai multe dataset-uri publice în format COCO, de pe Roboflow sau Kaggle.

3.2. Standardizarea și unificarea dataset-urilor
Dataset-urile descărcate au avut formate, clase și rezoluții diferite. Pentru a obține un dataset unificat, am efectuat:
  -uniformizarea claselor (toate clasele au fost redenumite în: Scratch, Crack, Inclusion, Bubble, EdgeChip, OK)
  -conversia tuturor etichetelor în COCO JSON
  -combinarea tuturor anotărilor într-un singur fișier COCO mare
  -mutarea tuturor imaginilor într-un singur spațiu comun

3.3. Curățarea imaginilor

Imaginile brute prezentau variații mari în format și calitate. În etapa de curățare au fost aplicate:
  -conversia la format RGB
  -redimensionarea imaginilor la 256×256 px
  -eliminarea duplicatelor
  -redenumirea uniformă a fișierelor
  -normalizarea valorilor pixelilor (0–1)

3.4. Împărțirea dataset-ului în seturi dedicate
Pentru antrenarea și evaluarea corectă a rețelei neuronale, dataset-ul a fost împărțit după standardele ML:
  -70% — train
  -15% — validation
  -15% — test

Împărțirea este stratificată, astfel încât fiecare tip de defect să fie reprezentat proporțional în fiecare subset.

3.5. Rezultatul final al dataset-ului
Dataset-ul final folosit pentru proiect conține:
  -imagini preprocesate (RGB, 256×256 px)
  -etichete standardizate pentru 6 clase
  -foldere organizate train/validation/test
  -fișier COCO final cu toate anotările combinate

Setul de date este acum compatibil cu TensorFlow/Keras și pregătit pentru etapa de antrenare a rețelei neuronale.
