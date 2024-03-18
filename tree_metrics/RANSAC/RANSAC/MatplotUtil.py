import matplotlib.pyplot as plt

#
#Takes an array of Point objects and return 2 lists of X and Y values
#
def create_list_from_points(points):
    list_x=[]
    list_y=[]
    for new_point in points:
        list_x.append(new_point.X)
        list_y.append(new_point.Y)
    return (list_x,list_y)


def plot_new_points_over_existing_points(points_existing,points_new,title,label_existing,label_new, out_file_path):
    oldlist_x,oldlist_y=create_list_from_points(points_existing)
    newlist_x,newlist_y=create_list_from_points(points_new)
    union_x=newlist_x+oldlist_x
    union_y=newlist_y+oldlist_y

    lst_limits=[min(union_x),max(union_x),min(union_y),max(union_y)]
    fig2 = plt.figure()
    plt.xlim(min(lst_limits)-3, max(lst_limits)+3)
    plt.ylim(min(lst_limits)-3, max(lst_limits)+3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(newlist_x, newlist_y,c="green",label=label_new)
    plt.scatter(oldlist_x, oldlist_y,c="blue",label=label_existing)
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.pause(1)
     
    fig2.savefig(out_file_path, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig2)
    
    pass

