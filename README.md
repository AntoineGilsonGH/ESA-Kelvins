# ESA Kelvins : prévoir le risque de collision d'un satellite

Travail sur le **Spacecraft Collision Avoidance Challenge** de l'Agence spatiale européenne (2019) : à partir des alertes de rapprochement reçues jusqu'à 2 jours avant le passage, prédire la probabilité de collision finale.

Ce dépôt contient pour l'instant la chaîne de données, la métrique officielle, les baselines de l'article de référence (reproduites au millième près), une validation croisée et une analyse exploratoire. Les modèles de ML et de deep learning sont la suite du travail (voir [Feuille de route](#feuille-de-route)).

---

## 1. Le problème

### Le contexte opérationnel

Le Space Debris Office de l'ESA surveille les satellites de l'agence (Sentinel, Swarm, Cryosat, Aeolus...) et ceux de partenaires. Quand le catalogue américain détecte qu'un objet (débris, étage de fusée, autre satellite) va passer près d'un de ces satellites, il émet un **CDM** (*Conjunction Data Message*). Un CDM contient la géométrie du rapprochement (distance de passage, vitesse relative), les incertitudes sur la position des deux objets (covariances), des informations sur la qualité de leur orbite et la **probabilité de collision** calculée.

Pendant la semaine qui précède le rapprochement (le **TCA**, *time of closest approach*), environ trois CDM par jour mettent à jour l'estimation. Un **événement** est donc une série temporelle de CDM. Les opérateurs manœuvrent typiquement si le risque dépasse 10⁻⁴ ou 10⁻⁵, et doivent décider **au moins 2 jours avant** le TCA.

### La question du challenge

> Avec les CDM disponibles 2 jours avant le TCA, peut-on prédire le risque du **dernier** CDM (émis moins d'un jour avant le TCA) ?

```
  jours avant TCA :   7 ......... 2 | 1 ...... 0
  CDM             :   x   x   x   x | ·   ·  y
                      entrées        caché   cible
```

- **Entrées** : tous les CDM émis au moins 2 jours avant le TCA (103 colonnes chacun).
- **Cible** : `risk` du dernier CDM, en log₁₀. −30 veut dire « négligeable ».
- Un événement est **à haut risque** si son risque final est ≥ −6 (10⁻⁶).
- Le test officiel ne retient que les événements avec au moins 2 CDM, un premier CDM au moins 2 jours avant le TCA et un dernier CDM moins d'un jour avant. Les mêmes règles sont appliquées au train (`kelvins.data.make_train`).

### La métrique

$$L = \frac{\text{MSE}_{HR}}{F_2}\qquad(\text{plus petit} = \text{meilleur})$$

- **F₂** : score F de la classification « risque final ≥ −6 », calculé sur tous les événements. β = 2, donc le rappel compte 4 fois plus que la précision : rater un vrai danger coûte bien plus cher qu'une fausse alerte.
- **MSE_HR** : erreur quadratique sur le log-risque, **uniquement sur les événements réellement à haut risque**. Toute prédiction sous −6 est d'abord remplacée par −6,001.

Conséquence : pour un événement qu'on pense sans danger, prédire −6,001 est toujours optimal ; tout l'enjeu est autour du seuil.

### Les données

| | Événements | CDM | Haut risque |
|---|---|---|---|
| Train brut | 13 154 | 162 634 | 365 (2,8 %) |
| Train éligible (règles du test) | 8 293 | 92 707 en entrée | **66 (0,8 %)** |
| Test officiel | 2 167 | 24 484 | **150 (6,9 %)** |

Source : [Zenodo 4463683](https://zenodo.org/records/4463683), licence CC-BY 4.0. Les données ne sont pas versionnées : `python scripts/download_data.py`.

### Ce qu'ont fait les participants (Uriot et al., 2020)

96 équipes, 862 soumissions. Résultats marquants :
- La baseline naïve **« le risque final = le dernier risque connu » (LRP)** fait L = 0,694. Seules **12 équipes sur 97** l'ont battue, et la meilleure (0,556) ne la bat que de 20 %.
- L'équipe gagnante (`sesc`) n'a **pas utilisé de ML** au final : des règles à seuils, réglées en partie par essais sur le classement public.
- La 3ᵉ équipe a utilisé un réseau siamois (Manhattan-LSTM) pour détecter les événements « anormaux » (bas → haut) parmi ceux sous le seuil.
- 65 % des équipes ont ignoré la dimension temporelle (dernier CDM + statistiques résumées).
- Après la compétition, les auteurs ont entraîné 5 000 modèles LightGBM : **un seul** a battu LRP à la fois sur le train et le test.

---

## 2. Résultats actuels

`python scripts/run_baselines.py`, qui calcule le score sur le test officiel avec un intervalle de confiance bootstrap à 95 %, et le score en validation croisée sur le train éligible (5 plis × 3 répétitions, stratifiés sur le haut risque).

| Modèle | Test L | IC 95 % | MSE_HR | F2 | FN | FP | CV L | Gain CV vs LRP | Article |
|---|---|---|---|---|---|---|---|---|---|
| Risque constant −5 (CRP) | 2,504 | [1,86 ; 3,34] | 0,679 | 0,271 | 0 | 2017 | 13,14 | −1 681 % | 2,5 |
| **Dernier risque connu (LRP)** | **0,694** | [0,46 ; 0,97] | 0,513 | 0,739 | 35 | 63 | 0,806 | 0 % | 0,694 |
| Règles de l'équipe gagnante | 0,582 | [0,39 ; 0,84] | 0,426 | 0,733 | 19 | 163 | 1,074 | −36 % | 0,569 |
| LightGBM sur le résidu, dernier CDM | 1,338 | [0,92 ; 1,83] | 0,884 | 0,660 | 52 | 44 | 1,300 | −53 % | |
| LightGBM sur le résidu, + historique | 2,738 | [1,81 ; 4,07] | 1,245 | 0,455 | 88 | 20 | 2,042 | −169 % | |

Lecture :
- **Les baselines officielles sont reproduites exactement** (CRP 2,50 ; LRP 0,694 / 0,513 / 0,739). Chaque règle de l'équipe gagnante prise seule reproduit le tableau 4 de l'article ; leur combinaison complète donne 0,582 au lieu de 0,569, l'article ne précisant pas entièrement comment elles se combinent.
- **Les règles gagnantes battent LRP sur le test mais perdent 36 % en validation croisée.** Elles poussent vers le haut les cas limites, ce qui paie sur un test qui contient 9 fois plus de hauts risques que le train. C'est de l'adaptation au test, pas une règle générale.
- **Le LightGBM « naïf » de l'article fait pire que LRP**, comme dans l'article. En régression, le modèle apprend que le risque baisse en moyenne, et fait donc passer sous le seuil des événements qui finiront dangereux (52 faux négatifs au lieu de 35).
- **L'intervalle de confiance est énorme** : [0,46 ; 0,97] pour LRP. Avec 150 événements positifs, un écart de quelques pourcents entre deux modèles n'est pas significatif.

---

## 3. Ce que montre l'analyse exploratoire

Notebook : [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

1. **Le dernier risque connu est de loin le meilleur prédicteur.** Sur le train éligible : 7 966 événements bas → bas, 47 haut → haut, 261 fausses alertes haut → bas, et seulement **19 surprises bas → haut**.
2. **Le changement de risque est bimodal** : soit il reste à peu près stable, soit il s'effondre vers −30 quand l'incertitude se resserre. Près du seuil, 53 à 61 % des événements s'effondrent et 1 à 2 % montent. Une régression qui moyenne ces deux issues est mal adaptée.
3. **Ce qui distingue les surprises** parmi les cas limites (−8 ≤ r₋₂ < −6) : une distance de passage plus petite (médiane 3,5 km contre 7,2 km), une distance de Mahalanobis deux fois plus faible et un `max_risk_estimate` plus élevé.
4. **Train et test ne se ressemblent pas** : 0,8 % contre 6,9 % de hauts risques, et une répartition des missions différente. Chaque `mission_id` est un satellite sur une orbite différente.
5. Les chasseurs de type `PAYLOAD` ne produisent presque jamais de haut risque (0 sur 1 007 dans le train).

---

## 4. Organisation du code

```
src/kelvins/
  config.py          seuils et constantes du challenge
  data.py            lecture, règles d'éligibilité, EventDataset (CDM d'entrée + risque final)
  metrics.py         métrique officielle, F2, MSE_HR, intervalle bootstrap
  features.py        variables du dernier CDM + résumés de l'historique
  validation.py      validation croisée répétée, toujours comparée à LRP sur les mêmes plis
  models/
    baselines.py     CRP, LRP, règles de l'équipe gagnante
    gbm.py           LightGBM qui corrige le dernier risque (expérience de l'article)
scripts/
  download_data.py   téléchargement depuis Zenodo
  run_baselines.py   évaluation de tous les modèles -> results/leaderboard.md
notebooks/01_eda.ipynb
tests/               métrique + reproduction des chiffres de l'article
```

Chaque modèle suit la même interface :

```python
from kelvins.data import make_train, make_test
from kelvins.metrics import report

class MonModele:
    def fit(self, ds):            # ds : EventDataset (ds.cdms, ds.y, ds.last, ds.latest_risk)
        ...
        return self
    def predict(self, ds):        # pd.Series de log10(risque), indexée par event_id
        ...

train, test = make_train(), make_test()
print(report(test.y, MonModele().fit(train).predict(test)))
```

Pour le comparer aux autres, il suffit de l'ajouter au dictionnaire `MODELS` de `scripts/run_baselines.py`.

## 5. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py      # 221 Mo depuis Zenodo
pytest -q                            # 12 tests, dont la reproduction des scores de l'article
python scripts/run_baselines.py      # ~1 min
```

Sous macOS, dans un dossier synchronisé par iCloud, Python 3.12+ peut ignorer l'installation éditable (fichier `.pth` marqué caché). Les scripts, le notebook et pytest ajoutent donc `src/` au chemin eux-mêmes.

---

## Feuille de route

L'idée est de monter en complexité, et de ne garder une étape que si elle bat LRP **en validation croisée et sur le test**, en regardant les intervalles de confiance.

**Étape 1, ML classique**
- [ ] Reformuler en deux sous-problèmes : un **classifieur** « le risque final sera-t-il ≥ −6 ? » (seuil de décision choisi pour maximiser F2), puis une valeur pour les positifs (LRP, ou une régression entraînée uniquement sur les hauts risques).
- [ ] Variables tirées de l'EDA : distance de passage, Mahalanobis, `max_risk_estimate`, type d'objet, tendance du risque sur les derniers CDM.
- [ ] Gérer le déséquilibre : poids de classe, focal loss, sous-échantillonnage des événements à −30.
- [ ] Calibrer les probabilités (Platt, isotonique) et regarder la courbe précision-rappel.
- [ ] Utiliser aussi les ~4 800 événements non éligibles du train pour apprendre la dynamique du risque (pré-entraînement, ou CDM intermédiaires comme cibles supplémentaires).

**Étape 2, deep learning**
- [ ] Modèle séquentiel (GRU / LSTM / petit transformer) sur la série de CDM, avec le temps avant TCA en entrée (séries irrégulières).
- [ ] Sortie probabiliste : mélange « s'effondre vers −30 » / « reste près de la dernière valeur », pour coller à la bimodalité.
- [ ] Comparer honnêtement au gradient boosting de l'étape 1 sur les mêmes plis.

**Étape 3, aller plus loin**
- [ ] Construire un découpage train/test plus sain (stratifié par mission) et mesurer la généralisation à une mission jamais vue.
- [ ] Quantifier l'incertitude de la prédiction (ensembles, prédiction conforme).

## Références

- T. Uriot, D. Izzo et al., [*Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition*](https://arxiv.org/abs/2008.03069), Astrodynamics (2021).
- [Page du challenge](https://kelvins.esa.int/collision-avoidance-challenge/) et [jeu de données (Zenodo)](https://zenodo.org/records/4463683).
- G. Acciarini et al., *Kessler: a machine learning library for spacecraft collision avoidance* (2021).
