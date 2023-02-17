
## Neuronale Netze: Zero to Hero

Ein Kurs über neuronale Netze, der ganz bei den Grundlagen beginnt. Der Kurs besteht aus einer Reihe von YouTube-Videos, in denen wir gemeinsam neuronale Netze programmieren und trainieren. Die Jupyter-Notizbücher, die wir in den Videos erstellen, werden dann hier im Verzeichnis [lectures](lectures/) festgehalten. Zu jeder Vorlesung gibt es auch eine Reihe von Übungen, die in der Videobeschreibung enthalten sind. (Dies kann sich zu etwas Ansehnlicherem entwickeln).

---

**Vorlesung 1: Die buchstabierte Einführung in neuronale Netze und Backpropagation: Aufbau von Mikrograd**

Backpropagation und Training von neuronalen Netzen. Setzt Grundkenntnisse in Python und eine vage Erinnerung an Kalkül aus der High School voraus.

- [YouTube-Video-Vorlesung](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Jupyter-Notizbuch-Dateien](lectures/micrograd)
- [micrograd Github Repo](https://github.com/karpathy/micrograd)

---

**Lecture 2: Die buchstabierte Einführung in die Sprachmodellierung: Aufbau von makemore**

Wir implementieren ein Bigram-Sprachmodell auf Zeichenebene, das wir in Folgevideos zu einem modernen Transformer-Sprachmodell, wie GPT, weiter komplexieren werden. In diesem Video liegt der Schwerpunkt auf (1) der Einführung von torch.Tensor und seinen Feinheiten und seiner Verwendung bei der effizienten Auswertung neuronaler Netze und (2) dem Gesamtrahmen der Sprachmodellierung, der Modelltraining, Sampling und die Auswertung eines Verlustes (z.B. die negative log Likelihood für die Klassifizierung) umfasst.

- [YouTube-Videovortrag](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Jupyter notebook files](lectures/makemore/makemore_part1_bigrams.ipynb)
- [makemore Github Repo](https://github.com/karpathy/makemore)

---

**Lecture 3: Aufbau von makemore Teil 2: MLP**

Wir implementieren ein mehrschichtiges Perceptron (MLP) Sprachmodell auf Zeichenebene. In diesem Video stellen wir auch viele Grundlagen des maschinellen Lernens vor (z.B. Modelltraining, Abstimmung der Lernrate, Hyperparameter, Evaluierung, Train/Dev/Test-Splits, Under/Overfitting, usw.).

- [YouTube-Video-Vorlesung](https://youtu.be/TCH_1BHY58I)
- [Jupyter-Notebook-Dateien](lectures/makemore/makemore_part2_mlp.ipynb)
- [makemore Github Repo](https://github.com/karpathy/makemore)

---

**Lecture 4: Aufbau von makemore Teil 3: Aktivierungen & Gradienten, BatchNorm**

Wir tauchen in die Interna von MLPs mit mehreren Schichten ein und untersuchen die Statistiken der Vorwärtspass-Aktivierungen, die Rückwärtspass-Gradienten und einige der Fallstricke, wenn sie falsch skaliert sind. Wir sehen uns auch die typischen Diagnosetools und Visualisierungen an, die Sie verwenden sollten, um den Zustand Ihres tiefen Netzwerks zu verstehen. Wir erfahren, warum das Training von tiefen neuronalen Netzen anfällig sein kann und stellen die erste moderne Innovation vor, die das Training erheblich erleichtert hat: Batch-Normalisierung. Residuale Verbindungen und der Adam-Optimierer bleiben bemerkenswerte ToDos für spätere Videos.

- [YouTube-Video-Vortrag](https://youtu.be/P6sfmUTpUmc)
- [Jupyter-Notizbuch-Dateien](lectures/makemore/makemore_part3_bn.ipynb)
- [makemore Github Repo](https://github.com/karpathy/makemore)

---

**Lecture 5: Aufbau von makemore Teil 4: Ein Backprop-Ninja werden**

Wir nehmen den 2-Schicht-MLP (mit BatchNorm) aus dem vorigen Video und führen ein Backpropagate durch, ohne die loss.backward() von PyTorch autograd zu verwenden. Das heißt, wir gehen rückwärts durch den Kreuzentropieverlust, die 2. lineare Schicht, tanh, Batchnorm, die 1. lineare Schicht und die Einbettungstabelle. Auf diesem Weg erhalten wir ein intuitives Verständnis dafür, wie Gradienten rückwärts durch den Berechnungsgraphen und auf der Ebene effizienter Tensoren fließen, nicht nur einzelner Skalare wie in micrograd. Dies trägt dazu bei, Kompetenz und Intuition dafür zu entwickeln, wie neuronale Netze optimiert werden und versetzt Sie in die Lage, moderne neuronale Netze selbstbewusst zu entwickeln und zu debuggen.

Ich empfehle Ihnen, die Übung selbst durchzuarbeiten, aber arbeiten Sie mit ihr im Tandem und wenn Sie nicht weiterkommen, pausieren Sie das Video und sehen Sie, wie ich die Antwort verrate. Dieses Video ist nicht dazu gedacht, einfach nur angeschaut zu werden. Die Übung ist [hier als Google Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing). Viel Glück :)

- [YouTube-Video-Vorlesung](https://youtu.be/q8SA3rM6ckI)
- [Jupyter notebook files](lectures/makemore/makemore_part4_backprop.ipynb)
- [makemore Github Repo](https://github.com/karpathy/makemore)

---

**Lecture 6: Aufbau von makemore Teil 5: Aufbau von WaveNet**

Wir nehmen den 2-Schicht-MLP aus dem vorherigen Video und bauen ihn mit einer baumähnlichen Struktur tiefer auf, um zu einer Architektur eines faltbaren neuronalen Netzwerks ähnlich dem WaveNet (2016) von DeepMind zu gelangen. In dem WaveNet Paper wird dieselbe hierarchische Architektur mit kausal dilatierten Faltungen (noch nicht behandelt) effizienter implementiert. Auf dem Weg dorthin bekommen wir einen besseren Eindruck davon, was torch.nn ist und wie es unter der Haube funktioniert und wie ein typischer Deep Learning-Entwicklungsprozess aussieht (viel Lesen von Dokumentation, Verfolgen von mehrdimensionalen Tensorformen, Wechsel zwischen Jupyter-Notebooks und Repository-Code, ...).

- [YouTube-Video-Vortrag](https://youtu.be/t3YJ5hKiMQ0)
- [Jupyter-Notizbuch-Dateien](lectures/makemore/makemore_part5_cnn1.ipynb)

---


**Lecture 7: Let's build GPT: von Grund auf, in Code, buchstabiert**.

Wir bauen einen Generatively Pretrained Transformer (GPT), in Anlehnung an das Papier "Attention is All You Need" und OpenAIs GPT-2 / GPT-3. Wir sprechen über Verbindungen zu ChatGPT, das die Welt im Sturm erobert hat. Wir beobachten, wie GitHub Copilot, selbst ein GPT, uns hilft, ein GPT zu schreiben (Meta :D!) . Ich empfehle, sich die früheren makemore Videos anzusehen, um sich mit dem autoregressiven Sprachmodellierungsrahmen und den Grundlagen von Tensoren und PyTorch nn vertraut zu machen, die wir in diesem Video als selbstverständlich voraussetzen.

- [YouTube-Videovortrag](https://www.youtube.com/watch?v=kCc8FmEb1nY). Alle anderen Links finden Sie in der Videobeschreibung.

---

Laufend...

**Lizenz**

MIT
