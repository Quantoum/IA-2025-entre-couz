import random
from game_manager import *
from random_agent import *

class Agent:

    def __init__(self, player):
        self.cnt = 0
        self.profondeur = 5
        self.historique = None
        self.lastKing = None
        self.transposition_table = {}  # Cache des évaluations
        self.move_consequences = {}
        self.ajustementProfTime = 0
        self.ajustementProfPiece = 0
        self.player = player
    
    def changeProf(self, prof):
        self.profondeur = prof
    
    def act(self, state, remaining_time):
        # #print("///////////////" + str(state.turn) + "///////////")
        if(len(state.actions()) == 1):
            return(state.actions()[0])
        
        if(remaining_time < 50):
            self.ajustementProfTime = - 3
        elif(remaining_time < 100):
            self.ajustementProfTime = - 2
        elif(remaining_time < 200):
            self.ajustementProfTime = - 1
        else:
            self.ajustementProfTime = 0
        cnt = len(state.pieces.values())

        if(cnt < 8):
            self.ajustementProfPiece = 4 
        elif(cnt < 10):
            self.ajustementProfPiece = 3 
        elif(cnt < 12):
            self.ajustementProfPiece = 2  
        elif(cnt < 15):
            self.ajustementProfPiece = 1
        else:
            self.ajustementProfPiece = 0
        
        #print(str(self.profondeur)  + " ")
        #print(str(self.ajustementProfTime) + " ")
        #print(str(self.ajustementProfPiece) + "\n")
        
        # if(state.can_create_general):
        #     self.ajustementProfPiece = self.ajustementProfPiece -1

        
        self.transposition_table.clear()
        self.move_consequences.clear()
        actions = state.actions()
        self.cnt +=1
        action = None
        maxaction = 1000
        minaction = -1000
        self.lastKing = state.can_create_king
        self.historique = state.history_boring_turn_hash
        actions = [a for a in actions if not self.is_useless_pawn_action(state, a)]
        map = self.init_id_map(state)

        actionRectifierPourRoi = actions
        if state.can_create_king and self.cnt > 6:
            actionRectifierPourRoi = []
            for possibleaction in actions:
                if(3*self.player in state.result(possibleaction).pieces.values()):
                    actionRectifierPourRoi.append(possibleaction)
        if self.cnt == 1 and self.player == -1:
            return state.actions()[17]
        elif self.cnt == 2 and self.player == -1:
            return state.actions()[10]
        elif self.cnt < 6 and self.player == -1:
            for i in range(0, 8):
                for j in range(0, 8):
                    if(i+j == 4):
                        if(not (i,j) in state.pieces and j <= i):
                            items = [a for a in actions if (a[0][0] + a[0][1] == 8 and a[1][0] + a[1][1] == 9 and a[0][0] == i and a[1][0] == i)]
                            if(len(items) == 0):
                                items = [a for a in actions if (a[0][0] + a[0][1] == 8 and a[1][0] + a[1][1] == 9 and a[0][0] == i+1 and a[1][0] == i)]
                            if(len(items) == 1):
                                return items[0]
                        if(not (i,j) in state.pieces and i <= j):
                            items = [a for a in actions if (a[0][0] + a[0][1] == 8 and a[1][0] + a[1][1] == 9 and a[0][1] == j and a[1][1] == j)]
                            if(len(items) == 0):
                                items = [a for a in actions if (a[0][0] + a[0][1] == 8 and a[1][0] + a[1][1] == 9 and a[0][1] == j+1 and a[1][1] == j)]
                            if(len(items) == 1):
                                return items[0]

            actions = [a for a in actions if a[0][0] + a[0][1] == 8 and a[1][0] + a[1][1] == 9 ]
            if(self.cnt == 5):
                return actions[-1]
            else:
                return actions[0]
        elif self.cnt == 1 and self.player == 1:
            return state.actions()[5]
        elif self.cnt == 2 and self.player == 1:
            return state.actions()[0]
        elif self.cnt < 6 and self.player == 1:
            for i in range(0, 8):
                for j in range(0, 8):
                    if(i+j == 8):
                        if(not (i,j) in state.pieces and j >= i):
                            items = [a for a in actions if (a[0][0] + a[0][1] == 5 and a[1][0] + a[1][1] == 4 and a[0][0] == i-1 and a[1][0] == i)]
                            if(len(items) == 0):
                                items = [a for a in actions if (a[0][0] + a[0][1] == 5 and a[1][0] + a[1][1] == 4 and a[0][0] == i and a[1][0] == i)]
                            if(len(items) == 1):
                                return items[0]
                        if(not (i,j) in state.pieces and i >= j):
                            items = [a for a in actions if (a[0][0] + a[0][1] == 5 and a[1][0] + a[1][1] == 4 and a[0][1] == j-1 and a[1][1] == j)]
                            if(len(items) == 0):
                                items = [a for a in actions if (a[0][0] + a[0][1] == 5 and a[1][0] + a[1][1] == 4 and a[0][1] == j and a[1][1] == j)]
                            if(len(items) == 1):
                                return items[0]
            actions = [a for a in actions if a[0][0] + a[0][1] == 5 and a[1][0] + a[1][1] == 4]
            if(self.cnt == 5):
                return actions[-1]
            else:
                return actions[0]
        else:
            for stateMax in actionRectifierPourRoi:
                #if(self.consequence_of_action(state, stateMax, map)):
                    resultMin = self.getMin(state.result(stateMax), self.profondeur + self.ajustementProfTime + self.ajustementProfPiece, minaction, self.update_id_map(map, stateMax))
                    resultMin -= (self.historique.count(state.result(stateMax)._hash()) * 100)
                    #if(self.historique.count(state.result(stateMax).precomputed_hash)) > 0:
                        #print("ok c'est bon mtn ")
                    if(resultMin > minaction):
                        minaction = resultMin
                        action = stateMax
        #for cle, valeur in self.move_consequences.items():
            #print(f"Clé : {cle}, Valeur : {valeur}")       
        if action != None:
            return action


        #print("debut")
        #print(self.eval(state))
        #print("fini")
        if len(actions) == 0:
            raise Exception("No action available.")
        return random.choice(actions)
    
    def eval(self, state):
        sum_reward = 0
        hadKing = False
        for valeurs in state.pieces.values():
            if(abs(valeurs) == 2):
                sum_reward += valeurs*3
            elif (abs(valeurs) == 3):
                sum_reward += valeurs*5
                if(valeurs*self.player > 0):
                    hadKing = True
            else:
                sum_reward += valeurs
        if(self.lastKing and not hadKing):
            sum_reward -= 100
        sum_reward = sum_reward*self.player
        
        return sum_reward
    
    def getMax(self, state, profondeur, maxActionUp, map):
        state_hash = state._hash()
        if (state_hash, profondeur) in self.transposition_table:
            return self.transposition_table[(state_hash, profondeur)]

        if(profondeur == 0):
            val = self.eval(state)
            self.transposition_table[(state_hash, profondeur)] = val
            return val
        
        maxaction = -1000
        actions = [a for a in state.actions() if not self.is_useless_pawn_action(state, a)]
        for stateMax in actions:
            #if(self.consequence_of_action(state, stateMax, map)):
                
                min = self.getMin(state.result(stateMax), profondeur-1, maxaction, self.update_id_map(map, stateMax))
                if(min > maxaction):
                    maxaction = min
                    if(maxaction >= maxActionUp):
                        break
        
        self.transposition_table[(state_hash, profondeur)] = maxaction

        return maxaction

    def getMin(self, state, profondeur, minactionUP, map):
        state_hash = state._hash()
        if (state_hash, profondeur) in self.transposition_table:
            return self.transposition_table[(state_hash, profondeur)]

        if(profondeur == 0):
            val = self.eval(state)
            self.transposition_table[(state_hash, profondeur)] = val
            return val
        
        minaction = 1000
        actions = [a for a in state.actions() if not self.is_useless_pawn_action(state, a)]

        for stateMin in actions:
            #if(self.consequence_of_action(state, stateMin, map)):

                min = self.getMax(state.result(stateMin), profondeur-1, minaction,  self.update_id_map(map, stateMin))
                if(min < minaction):
                    minaction = min
                    if(min <= minactionUP):
                        break
        self.transposition_table[(state_hash, profondeur)] = minaction

        return minaction
    
    def is_useless_pawn_action(self, state, action):
        from_pos = action[0]
        piece = state.pieces.get(from_pos)

        if piece is None or abs(piece) >= 2:
            return False  # Roi ou vide : on garde

        if piece * self.player <= 0:
            return False  # Pas notre pion

        x, y = from_pos

        for pos, other_piece in state.pieces.items():
            x2 = pos[0]
            y2 = pos[1]

            if (x2, y2) == (x, y):
                continue
            if other_piece * self.player > 0 and (x2 == x + (1*self.player) or x2 == x) and (y2 == y +(1*self.player)  or y2 == y):
                if self.has_valid_actions(state, (x2, y2)):

                    return False  # On est "derrière" un autre pion de notre équipe

        return False

    def has_valid_actions(self, state, pos):
        actions = state.actions()  # Récupère toutes les actions possibles dans le jeu
        for action in actions:
            if action.start == pos:  # Si l'action commence à la position donnée
                return True  # Action valide trouvée
        return False

    def consequence_of_action(self, actualstate, action, map):
        a, b = action[0]
        grid = self.id_map_to_grid(map, [7,8])
        id = grid[a][b]    
        followingState = actualstate.result(action)
        consequence = self.eval(followingState.result(followingState.actions()[0])) - self.eval(followingState)

        if (action[0] , action[1], id) in self.move_consequences and self.move_consequences[(action[0], action[1], id)] == consequence:
            return False
        else:
            self.move_consequences[(action[0], action[1], id)] = consequence
            return True
    
    def init_id_map(self, state):
        id_map = {}
        next_pos_id = 1
        next_neg_id = -1
        for pos, val in sorted(state.pieces.items()):
            if val > 0:
                id_map[pos] = next_pos_id
                next_pos_id += 1
            elif val < 0:
                id_map[pos] = next_neg_id
                next_neg_id -= 1
        return id_map
    
    def update_id_map(self, id_map, action):
        new_id_map = dict(id_map)

        # Le pion se déplace
        moving_id = new_id_map.pop(action.start)
        new_id_map[action.end] = moving_id

        # Les pions capturés disparaissent
        for removed_pos in action.removed:
            if removed_pos in new_id_map:
                del new_id_map[removed_pos]

        return new_id_map
     
    def id_map_to_grid(self, id_map, dim):
        grid = [[0 for _ in range(dim[1])] for _ in range(dim[0])]
        for (i, j), v in id_map.items():
            grid[i][j] = v
        return grid