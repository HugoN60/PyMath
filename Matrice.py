class Matrice():
    def __init__(self, matrice : list):
        if not all(isinstance(row, list) for row in matrice):
            raise TypeError("A matrix must be a List[List[]]")
        assert Matrice.isMatrice(matrice), "All the row of a matrix must have the same length"
        self.__matrice = matrice
        self.__dim = [len(self.__matrice), len(self.__matrice[0])]
        self.__max = max([max(l) for l in matrice])

    def __str__(self):
        result = []
        taille = len(str(self.__max))
        for idx, ligne in enumerate(self.__matrice):
            valeurs = " ".join(f"{v:^{taille}}" for v in ligne)
            if len(self.__matrice) == 1:
                return f"[{valeurs}]\n"
            if idx == 0:
                result.append(f"⎡{valeurs}⎤")
            elif idx == len(self.__matrice) - 1:
                result.append(f"⎣{valeurs}⎦")
            else :
                result.append(f"⎢{valeurs}⎥")
        return "\n".join(result)
    
    def __repr__(self):
        dim = "dim: " + "x".join(f"{v}" for v in self.__dim)
        return self.__str__() + "\n" + dim
    
    def __eq__(self, o):
        if o is None:
            return False
        if not isinstance(o, Matrice):
            return False
        return self.__matrice == o.__matrice
    
    def __add__(self, o):
        if o is None:
            raise TypeError("The argument is None")
        if not isinstance(o, Matrice):
            raise TypeError(f"{o} is not a Matrix")
        if self.__dim != o.__dim:
            raise ValueError("Dim do not match")
        result = [[x + y for x, y in zip(ligne_a, ligne_b)] 
                for ligne_a, ligne_b in zip(self.__matrice, o.__matrice)]
        return Matrice(result)
    
    def __mul__(self, o):
        if o is None:
            raise TypeError("The argument is None")
        if isinstance(o, (int, float)):
            result = [[x*o for x in ligne] for ligne in self.__matrice]
            return Matrice(result)
        elif isinstance(o, Matrice):
            if(self.__dim[1] != o.__dim[0]):
                raise ValueError("Dim do not match")
            result = [[0 for x in range(o.__dim[1])] for y in range(self.__dim[0])]
            for i in range(self.__dim[0]):
                for j in range(o.__dim[1]):
                    for k in range(self.__dim[1]):
                        result[i][j] += self.__matrice[i][k] * o.__matrice[k][j]
            return Matrice(result) 
    
    def __rmul__(self, o):
        return self.__mul__(o)           
                
    @staticmethod
    def isMatrice(m):
        tailleL = len(m[0])
        for i in range(1, len(m)):
            if len(m[i]) != tailleL:
                return False
        return True

    
tab = [[1, 2, 3, 4],
       [5, 6, 7, 8],
       [9, 10, 11, 12]]

tab2 = [[1, 2, 3, 4],
       [5, 6, 7, 8],
       [9, 10, 11, 12]]

tab3 = [[1, 4, 3, 4],
       [5, 6, 7, 8],
       [9, 10, 11, 12],
       [13, 14, 15, 16]]

string = "foo"

m = Matrice(tab)
m2 = Matrice(tab2)
m3 = Matrice((tab3))
print(3*m)