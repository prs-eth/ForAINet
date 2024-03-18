

class Point:
    """Represents a point in 2 dimensional space"""
    idcounter=0
    def __init__(self, x,y):
        self.X=x
        self.Y=y
        self.ID=Point.idcounter+1
        Point.idcounter+=1
    def __str__(self):
        s="ID=%d x='%d' y='%d'" % (self.ID,self.X,self.Y)
        return s

    def __repr__(self):
        s="ID=%d ('%d','%d')" % (self.ID,self.X,self.Y)
        return s

    def __hash__(self):
        return self.ID
    #
    #Computes the euclidean distance between the 2 points
    #
    @staticmethod
    def euclidean_distance(point1,point2):
        squared=(point1.X-point2.X)**2 + (point1.Y-point2.Y)**2
        r=squared**0.5
        return r

