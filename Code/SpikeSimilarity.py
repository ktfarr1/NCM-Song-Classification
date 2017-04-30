import numpy as np

def spike_similarity(t_a, t_b):
    w = {"add":1.0, "delete":1.0, "move":1.0/400}
    #if diff is negative, we need to net remove items from a to transform it into b
    diff = np.count_nonzero(t_b) - np.count_nonzero(t_a)
    mask = np.invert(np.equal(t_a, t_b))
    score = 0
    score = decision_function(t_a,t_b,mask,diff,w,score)
    return score

def decision_function(t_a,t_b,mask,net_diff,w,score):
    # while (net_diff != 0 & True in mask):
    # & (True in mask)
    while((net_diff != 0) | (True in mask)):
        next_difference = np.random.choice(np.where(mask == True)[0])
        # next_difference = np.where(mask == True)[0][0]
        right, left = search_radius(next_difference,w,t_a.shape[0])
        if(t_a[next_difference]>t_b[next_difference]):
            l_move = check_path(t_a,next_difference,left,mask,0)
            r_move =check_path(t_a,next_difference,right,mask,1)
            if((len(l_move)==0 ) & (len(r_move)==0)):
                delete(t_a,next_difference)
                net_diff = net_diff+1
                mask = np.invert(np.equal(t_a, t_b))
                score = score+ (1*w["delete"])
                continue
            else:
                #check some shit
                if (len(l_move) > len(r_move)):
                    distance = abs(l_move[1]-l_move[0])
                    move(t_a,l_move[0],l_move[1])
                    score = score + (distance * w["move"])
                    mask = np.invert(np.equal(t_a, t_b))
                    continue
                else:
                    distance = abs(r_move[1]-r_move[0])
                    move(t_a,r_move[0],r_move[1])
                    score = score + (distance * w["move"])
                    mask = np.invert(np.equal(t_a, t_b))
                    continue


        else:
            l_move = check_path(t_a,next_difference,left,mask,0)
            r_move = check_path(t_a,next_difference,right,mask,1)
            if((len(l_move)==0) & (len(r_move)==0)):
                add(t_a,next_difference)
                net_diff = net_diff-1
                mask = np.invert(np.equal(t_a, t_b))
                score = score+(1*w["add"])
                continue
            else:
                if (len(l_move) > len(r_move)):
                    distance = abs(l_move[1]-l_move[0])
                    move(t_a,l_move[0],l_move[1])
                    score = score + (distance * w["move"])
                    mask = np.invert(np.equal(t_a, t_b))
                    continue
                else:
                    distance = abs(r_move[1]-r_move[0])
                    move(t_a,r_move[0],r_move[1])
                    score = score + (distance * w["move"])
                    mask = np.invert(np.equal(t_a, t_b))
                    continue

        # return right,left
        # return decision_function(t_a,t_b,w,diff+1,score+1)
    else:
        return score

def search_radius(a_index,weight,length):
    l_end = max(0, int(a_index - 1/weight["move"]))
    r_end = min(length,int(1 + a_index + 1/weight["move"]))
    r = [a_index+1,r_end]
    l = [l_end,a_index]
    return r, l

def add(a,a_index):
    a[a_index] = 1

def delete(a,a_index):
    a[a_index] = 0

def move(a,origin,dest):
    o = a[origin]
    d = a[dest]
    a[origin]=d
    a[dest] = o

def check_path(t_a,current,search_radius,mask,direction):
    first_one = np.nonzero(t_a[search_radius[0]:search_radius[1]])[0]
    if (first_one.shape[0] > 0):
        search_radius[direction] = np.arange(search_radius[0],search_radius[1])[first_one[0]]
    open_slots = np.where(mask[search_radius[0]:search_radius[1]]==True)[0]
    open_slots = np.arange(search_radius[0],search_radius[1])[open_slots]
    if (open_slots.shape[0]>0):
        for move in open_slots:
        # for move in np.nditer(open_slots):
            if (t_a[move] != t_a[current]):
                return np.asarray([current,move,direction])
    return []

