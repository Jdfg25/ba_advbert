# Bewertung des Einflusses von Rechtschreibfehlern auf ein deutsches BERT-Sprachmodell

Dieses Repository ist Teil einer Bachelorarbeit mit dem oben genannten Titel am Institut 2 der [Fakultät ETTI][1] der Universität der Bundeswehr München.  
Hierbei soll mithilfe des Modells [*bert-base-geman-uncased*][5] und des Datensatzes [Wikipedia in der Konfiguration 20200501.de][2] untersucht werden, inwiefern BERT durch das Training auf einen fehlerbehafteten Datensatz robust gegenüber dem Einfluss von Rechtschreibfehlern gemacht werden kann.

## Einrichtung

### Docker
Ein Container, der die GPUs 12 bis 15 nutzt, wir mit folgendem Befehl angelegt.
```
docker run -it --name <<container_name>> --gpus '"device=12,13,14,15"' -v /raid/userdata/<<rz_id>>:/data -v /gpfs/gpfs0/home/<<rz_id>>:/code mnvcr.io/nvidia/pytorch:20.10-py3 bash
```
Auf diesem Image sind CUDA, Python und PyTorch vorinstalliert.  
Alle Dateipfade sind auf diese Docker Umgebung ausgelegt und müssen für andere Arbeitsumgebungen angepasst werden.  
Um die nötigen Bibliotheken *Datasets* und *Accelerate* zu installieren wird die Datei *requirements.txt* verwendet.
```
pip install -r requirements.txt
```
Für eine korrekte Ausführung des Skripts *mod_run_mlm_no_trainer.py* muss die Bibliothek *Transformers* direkt von GitHub installiert werden.
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

### Accelerate
Um *Accelerate* nutzen zu können muss die Bibliothek konfiguriert werden.
```
accelerate config
```
Es folgen einige Fragen.
```
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0  
Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): 1
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
How many processes in total will you use? [1]: 4
Do you wish to use FP16 (mixed precision)? [yes/NO]:
```
Schließlich kann getestet werden ob alles korrekt läuft. Dies funktioniert allerdings nur bei 2, 4 und 8 GPUs.
```
accelerate test
```

## Skripte

### GenerateDataset.py
Das Skript steuert das Laden des Datensatzes und den Aufruf der weiteren Funktionen.  
Falls der Datensatz noch nicht lokal gespeichert ist wird er aus dem Netz geladen. Danach wird die Funktion *clean_dataset()* aus CleanDataset.py aufgerufen. 
Optional wird zusätzlich, gesteuert durch das Argument *insert_typos*, die Funktion *insert_typos()* aus InsertTypos.py aufgerufen.  
Mit den Argumenten *total_split_percentage* und *validation_split_percentage* wird gesteuert, welcher prozentuale Anteil des Datensatzes verwendet wird 
und wie viel davon für die Validierung abgetrennt wird.  
Ein Aufruf für 80% der Daten und das Einfügen von Rechtschreibfehlern sieht folgendermaßen aus:  
```
python GenerateDataset.py \ 
--total_split_percentage 80 \ 
--insert_typos 1
```

### CleanDataset.py
Die Artikel im Datensatz werden nach den Kapiteln *Weblinks* und *Literatur* durchsucht und selbige werden entfernt um das Pre-Training zu optimieren.

### InsertTypos.py
Mit der aus GenerateDataset.py übergebenen Wahrscheinlichkeit *true_prob* werden in den einzelnen Zeichen der Strings Rechtschreibfehler eingefügt. 
Diese basieren auf dem Paper [*Adv-BERT*][6] und bestehen aus dem Einfügen, Löschen und Austauschen eines Zeichens, sowie dem Vertauschen zweier Zeichen.  
Gesteuert durch das Argument *exclude* wird entweder jeder Artikel mit Fehlern durchsetzt oder nur zwei Drittel. In letzterem Fall werden die restlichen Daten nicht modifiziert.

### mod_run_mlm_no_trainer.py
Dieses Skript basiert auf [run_mlm_no_trainer.py][3] aus der [HuggingFace Transformers][4] Bibliothek, welches das Training eines Transformers mithilfe von Masked Language Modelling realisiert.  
Bei Aufruf können verschiedene Argumente übergeben werden, um den Trainingsbalauf zu konfigurieren.  

| Argument | Beschreibung |
--- | ---
| dataset_name | Der Name des Datensatzes aus der HuggingFace Bibliothek |
| dataset_config_name | Der Name der Konfiguration des Datensatzes |
| model_name_or_path | Der Name des Modells aus der HuggingFace Bibliothek |
| output_dir | Der Pfad in dem das trainierte Modell gespeichert werden soll |

Nachfolgend sind die zusätzlichen Funktionen der modifizierten Version gegenüber dem Original beschrieben.  
Mit *insert_typos* wird entweder der bereinigte oder der fehlerbehaftete Datensatz geladen, welcher mit GenerateDataset.py erstellt wurde. Ist nichts anderes angegeben wird durch dieses Argument auch der Pfad festgelegt in dem das trainierte Modell gespeichert wird.  
*train_pretrained* legt fest, ob nur die rohe Konfiguration des Modells als Basis genutzt wird, d.h. von Grund auf neu trainiert wird oder ob auch die vortrainierten Gewichte geladen werden. Wird ein lokales Modell geladen muss im Argument *tokenizer_name* zusätzlich angegeben werden, welcher Tokenizer genutzt werden soll.  
Zudem werden die Metriken Loss und Accuracy von Training und Evaluation in eine Textdatei gespeichert. Mithilfe von PlotLoss.py kann er als Graph dargestellt und gespeichert werden.

Ein beispielhafter Aufruf sieht folgendermaßen aus:
```
python mod_run_mlm_no_trainer.py \
--dataset_name wikipedia \
--dataset_config_name 20200501.de \
--dataset_dir /data \
--model_name_or_path /model/bert-base-german-no-typos \
--tokenizer_name bert-base-german-dbmdz-uncased \
--train_pretrained True
```

### Evaluate.py
Um die Leistungsfähigkeit des trainierten Modells zu beurteilen muss es das fehlende Wort in 10 verschiedenen Sätzen bestimmen.  
Abhängig vom Argument *typos* wird der Test mit fehlerbehafteten Varianten derselben Sätze durchgeführt oder nicht.  
Als Ausgabe erhält man für jeden Satz die 5 wahrscheinlichsten Wörter mitsamt ihrer Scores.


[1]: https://www.unibw.de/etti
[2]: https://huggingface.co/datasets/wikipedia#20200501de
[3]: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
[4]: https://huggingface.co/transformers/
[5]: https://huggingface.co/dbmdz/bert-base-german-uncased
[6]: https://arxiv.org/abs/2003.04985
