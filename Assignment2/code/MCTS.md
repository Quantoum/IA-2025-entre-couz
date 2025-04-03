# Explication du MCTS

### Documentation
[Documentation de MCTS avec UCT](https://ai-boson.github.io/mcts/) \
[Super github repo](https://github.com/brilee/python_uct/tree/master?tab=readme-ov-file) \
[Chouette essai sur le MCTS](https://www.moderndescartes.com/essays/deep_dive_mcts/)

Un court résumé du code mcts, afin de bien tout comprendre :))

Ici, le MCTS se déroule en 4 phases:
1. **Selection** \
    L'idée c'est de toujours choisir le meilleur enfant jusqu'à ce qu'on rejoigne une feuille. Pour ça, on utilise la formule d'UCT:
    (il y a une autre formule avec ln de Ni)
    ```math
        \frac{w_i}{n_i} +c\sqrt{\frac{ln(N_i)}{ni}}
    ```
    où : \
    $ w_i $ : nombre de wins au i-ème move \
    $ n_i $: nombre de simulations au i-ème move\
    $ c $ : paramètre d'exploration

2. **Expansion** \
    On arrive dans cette phase lorsqu'on ne peut plus appliquer UCT pour trouver la node successeur. On étends l'arbre en ajoutant tous les états possibles à partir de la feuille.
3. **Simulation** \
    (ou Rollout) Après l'expansion, l'algorithme choisi une node arbitrairement, et simule le jeu à partir de ce node (donc d'un état de jeu), jusqu'à obtenir une fin de jeu.
    * Light Play Out : Les nodes sont choisies aléatoirement
    * Heavy Play Out : Les nodes sont choisies à l'aide d'une heuristique ou d'une fonction d'évaluation
4. **Backpropagation** \
    Une fois que l'algorithme atteint la fin du jeu, il évalue l'état afin de savoir quel joueur a gagné. Il augmente donc le score de visite de tous les nodes en remontant.