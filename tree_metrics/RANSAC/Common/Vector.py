from .Point import Point

class Vector(object):
    """description of class"""
    def __init__(self,x:float,y:float):
        self.__x=x
        self.__y=y
        self.__length=-1
        self.__unitvector=None #the unit vector representation of this vector
        pass

    @property
    def X(self):
        return self.__x

    # @X.setter
    # def X(self,value):
    #     self.__x=value

    @property
    def Y(self):
        return self.__y

    # @Y.setter
    # def Y(self,value):
    #     self.__y=value

    @property
    def Length(self):
        if (self.__length != -1):
            return self.__length
        self.__length= (self.__x**2 + self.__y**2)**0.5
        return self.__length

    @property
    def UnitVector(self):
        if (self.__unitvector !=None):
            return self.__unitvector
        length=self.Length
        self.__unitvector = Vector(self.X/length, self.Y/length)
        return self.__unitvector
    
    def __repr__(self):
        return f"X={self.X} Y={self.Y}"
    #
    #Creates a vectory from point1 to point2
    #
    @classmethod
    def create_vector_from_2points(cls,point1:Point , point2:Point):
        v=Vector(point2.X-point1.X,point2.Y-point1.Y)
        return v

    @classmethod
    def dot_product(cls,vector1 , vector2):
        dot=(vector1.X * vector2.X)  + (vector1.Y*vector2.Y)
        return dot

    

