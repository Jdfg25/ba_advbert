# Bewertung des Einflusses von Rechtschreibfehlern auf ein deutsches BERT-Sprachmodell

Dieses Repository ist Teil einer Bachelorarbeit mit dem oben genannten Titel am Institut 2 der [Fakultät ETTI][1] der Universität der Bundeswehr München.  
Hierbei soll mithilfe des Modells [*bert-base-geman-uncased*][5] und des Datensatzes [Wikipedia in der Konfiguration 20200501.de][2] untersucht werden, inwiefern BERT durch das Training auf einen fehlerbehafteten Datensatz robust gegenüber dem Einfluss von Rechtschreibfehlern gemacht werden kann.  
Alle Dateipfade beziehen sich auf die Docker Umgebung in welchem die bisherige Arbeit durchgeführt wurde. Sie müssen für andere Arbeitsumgebungen eventuell angepasst werden.

## GenerateDataset.py
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

## CleanDataset.py
Die Artikel im Datensatz werden nach den Kapiteln *Weblinks* und *Literatur* durchsucht und selbige werden entfernt um das Pre-Training zu optimieren.

## InsertTypos.py
Mit der aus GenerateDataset.py übergebenen Wahrscheinlichkeit *true_prob* werden in den einzelnen Zeichen der Strings Rechtschreibfehler eingefügt. 
Diese basieren auf dem Paper [*Adv-BERT*][6] und bestehen aus dem Einfügen, Löschen und Austauschen eines Zeichens, sowie dem Vertauschen zweier Zeichen.  
Gesteuert durch das Argument *exclude* wird entweder jeder Artikel mit Fehlern durchsetzt oder nur zwei Drittel. In letzterem Fall werden die restlichen Daten nicht modifiziert.

## mod_run_mlm_no_trainer.py
Dieses Skript basiert auf [run_mlm_no_trainer.py][3] aus der [HuggingFace Transformers][4] Bibliothek, welches das Training eines Transformers mithilfe von Masked Language Modelling realisiert.  
Nachfolgend sind die zusätzlichen Funktionen gegenüber dem Original beschrieben.  
Mit *insert_typos* wird entweder der bereinigte oder der fehlerbehaftete Datensatz geladen, welcher mit GenerateDataset.py erstellt wurde. Ist nichts anderes angegeben wird durch dieses Argument auch der Pfad festgelegt in dem das trainierte Modell gespeichert wird.  
*train_pretrained* legt fest, ob nur die rohe Konfiguration des Modells als Basis genutzt wird, d.h. von Grund auf neu trainiert wird oder ob auch die vortrainierten Gewichte geladen werden. Wird ein lokales Modell geladen muss im Argument *tokenizer_name* zusätzlich angegeben werden, welcher Tokenizer genutzt werden soll.  
Zudem wird der Loss von Training und Evaluation in eine Textdatei gespeichert. Mithilfe von PlotLoss.py kann er als Graph dargestellt und gespeichert werden. 
Ein Aufruf könnte so aussehen:
```
python mod_run_mlm_no_trainer.py \
--dataset_name wikipedia \
--dataset_config_name 20200501.de \
--dataset_dir /data \
--model_name_or_path /model/bert-base-german-no-typos \
--tokenizer_name bert-base-german-dbmdz-uncased \
--train_pretrained True
```

## Evaluate.py
Um die Leistungsfähigkeit des trainierten Modells zu beurteilen muss es das fehlende Wort in 10 verschiedenen Sätzen bestimmen.  
Abhängig vom Argument *typos* wird der Test erneut mit fehlerbehafteten Varianten derselben Sätze durchgeführt.


[1]: https://www.unibw.de/etti
[2]: https://huggingface.co/datasets/wikipedia#20200501de
[3]: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
[4]: https://huggingface.co/transformers/
[5]: https://huggingface.co/dbmdz/bert-base-german-uncased
[6]: https://arxiv.org/abs/2003.04985
