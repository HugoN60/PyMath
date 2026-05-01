class Matrice():
    def __init__(self, matrice : list):
        if not all(isinstance(row, list) for row in matrice):
            raise TypeError("A matrix must be a List[List[]]")
        assert Matrice.isMatrice(matrice), "All the row of a matrix must have the same length"
        self.__matrice = [[float(x) for x in ligne] for ligne in matrice]
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
        return f"Matrice({self.__matrice})" 
    
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

        
    def inverse(self):
        """
        Computes the inverse matrix using the Gaussian elimination.

        Returns
        -------
        Matrice
            A new Matrice instance representing the inverse.

        Raises
        ------
        ValueError
            If the matrix is not square.
        ValueError
            if the inverse of the matrix doesn't exist
        """
        if self.__dim[0] != self.__dim[1]:
            raise ValueError("The matrix must be a square matrix")
        matrice = [ligne[:] for ligne in self.__matrice]
        identite = Matrice.identite(self.__dim).__matrice
        for pivot in range(self.__dim[0]):
            for li in range(self.__dim[0]):
                if li!=pivot:
                    piv = matrice[pivot][pivot]
                    act = matrice[li][pivot]
                    matrice[li] = [(piv * x) - (act * y) for x, y in zip(matrice[li], matrice[pivot])]
                    identite[li] =  [(piv * x) - (act * y) for x, y in zip(identite[li], identite[pivot])]
        for p in range(self.__dim[0]):
            nb = matrice[pivot][pivot]
            if nb == 0:
                raise ValueError("The inverse of the matrix doesn't exist ")
            identite[p] = [x / nb for x in identite[p]]
        return Matrice(identite)

                
    @staticmethod
    def isMatrice(m):
        """
        Return if m can become a matrix or not.

        Parameters
        ----------
        m : List[List[]]
            A List of List.

        Returns
        -------
        bool
            True if m can be a matrix, False otherwise.
        """
        tailleL = len(m[0])
        for i in range(1, len(m)):
            if len(m[i]) != tailleL:
                return False
        return True
    
    @staticmethod
    def identite(dim):
        """
        Computes the identity matrix for a given dimension.

        Parameters
        ----------
        dim : List[int]
            A List[int] which contains 2 int.

        Returns
        -------
        Matrice
            A new Matrice instance representing the identity.

        Raises
        ------
        ValueError
            If dim has less or more than 2 int.
        ValueError
            if dim does not contain only int.
        """
        if len(dim) != 2:
            raise ValueError("Dim argument must be a list of 2 element")
        if not all(isinstance(dim[i], int) for i in range(2)):
            raise ValueError("Dim argument must contain integers")
        matrice = [[0 for col in range(dim[1])] for li in range(dim[0])]
        for i in range(min(dim)):
            matrice[i][i] = 1
        return Matrice(matrice)
            
        

    
tab = [[1, 2, 3, 4],
       [5, 6, 7, 8],
       [9, 10, 11, 12]]

tab2 = [[1, 2, 3, 4],
       [5, 6, 7, 8],
       [9, 10, 11, 12]]

tab3 = [[1]]


m = Matrice(tab)
m2 = Matrice(tab2)
m3 = Matrice((tab3))
print(repr(Matrice.identite([3, 3])))
