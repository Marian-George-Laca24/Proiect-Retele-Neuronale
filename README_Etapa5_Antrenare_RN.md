# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

Disciplina: Rețele Neuronale  
Instituție: POLITEHNICA București – FIIR  
Student: Laca Marian-George 
Link Repository GitHub: https://github.com/Marian-George-Laca24/Proiect-Retele-Neuronale.git
Data: 16/12/2025


Antrenarea Modelului RN (Nivel 1 – Obligatoriu)

Modelul de rețea neuronală definit în Etapa 4 a fost antrenat efectiv folosind datasetul curent de imagini cu defecte pe sticlă, organizat în 5 clase:
Bubble, Crack, Inclusion, Scratch, OK.

Antrenarea s-a realizat folosind transfer learning cu arhitectura MobileNetV2, cu straturile convoluționale de bază înghețate și antrenarea doar a capului de clasificare.

⚙️ Hiperparametri utilizați și justificare

Hiperparametru  	Valoare	        Justificare
Optimizer   	    Adam	        Optimizator adaptiv, stabil pentru CNN-uri cu transfer learning
Learning rate	    0.001	        Valoare standard care asigură convergență rapidă fără instabilitate
Batch size  	    32	            Compromis între stabilitatea gradientului și consumul de memorie
Număr epoci 	    20	            Suficient pentru convergență pe dataset-ul curent
Loss function	    Categorical     Crossentropy	Potrivit pentru clasificare multi-clasă
Activare hidden	    ReLU	        Introduce non-linearitate eficientă
Activare output	    Softmax	        Produce distribuție de probabilități pe clase

📊 Rezultate obținute pe setul de test

Evaluarea a fost realizată pe setul de test separat, folosind scriptul evaluate.py.

Metrici obținute:

Accuracy: 98.41%

F1-score (macro): 0.93

Metricile au fost salvate automat în:

results/test_metrics.json

⚠️ Observație importantă privind performanța ridicată

Deși metricile obținute sunt foarte ridicate, acestea trebuie interpretate în contextul datasetului curent.

Datasetul utilizat este compus din imagini provenite din surse publice multiple, având:

imagini relativ curate,

defecte vizuale bine delimitate,

variație redusă a condițiilor de iluminare și perspectivă.

Acest lucru poate conduce la o supraestimare a performanței reale în condiții industriale.
În etapa următoare, datasetul va fi refăcut de la zero, folosind date originale capturate în condiții reale de producție, pentru o evaluare mai realistă a performanței modelului.

👉 Modelul este însă corect antrenat, evaluat și integrat, conform cerințelor Etapei 5 – Nivel 1.

📁 Istoricul Antrenării Modelului

Pe durata antrenării modelului, toate valorile relevante pentru fiecare epocă au fost salvate automat într-un fișier de tip CSV, pentru asigurarea trasabilității și posibilitatea de analiză ulterioară.

📄 Fișier:

results/training_history.csv

Conținutul fișierului

Fișierul training_history.csv conține, pentru fiecare epocă de antrenare, următoarele informații:

epoch – numărul epocii

loss – valoarea funcției de pierdere pe setul de antrenare

accuracy – acuratețea pe setul de antrenare

val_loss – pierderea pe setul de validare

val_accuracy – acuratețea pe setul de validare

Aceste date permit:

verificarea convergenței modelului,

identificarea fenomenelor de overfitting sau underfitting,

reproducerea și compararea rezultatelor în antrenări viitoare.

Crearea fișierului

Fișierul a fost generat automat de scriptul:

src/neural_network/train.py


în momentul finalizării procesului de antrenare, folosind istoricul returnat de metoda model.fit() din Keras.

Utilizare ulterioară

Datele din training_history.csv pot fi utilizate pentru:

generarea graficelor de tip loss vs epoch și accuracy vs epoch,

comparații între mai multe antrenări ale modelului,

documentarea evoluției performanței în etapele viitoare ale proiectului.



🖥️ Integrarea Modelului Antrenat în UI

Modelul antrenat (models/trained_model.h5) a fost integrat într-o aplicație Streamlit, care permite:

încărcarea unei imagini de către utilizator,

preprocesarea automată,

inferență reală cu modelul antrenat,

afișarea clasei prezise și a scorului de încredere.

O demonstrație vizuală a inferenței reale este disponibilă în:

docs/screenshots/inference_real.png
