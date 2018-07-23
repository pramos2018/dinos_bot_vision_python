'''
================================================================================================
DinoBot's Vision:
Author: Paulo Ramos
Date: June 2018.
================================================================================================
DinoBot Vision Idea:
1) Detect the screen and Display:
- Obstacles, sizes, positions and velocity;
- Dino's jump and distance moved;
2) Use DinoBot's vision to tringger the jumps
3) Use Machine Learning to train and improve Dino's performance
    - Output (1 = Jump, 0 = Do not Jump)
    - Features: Object Distance, Size and Velocity
    Goal: To use Machine Learning to find a correlation between Distance, Size and Velocity
    Adjust the weights to get the best correlation betweeen the variables.
    Note: The weights might be zero.
    Formula: factor = weight#0 + distance * weight#1 + size * weight#2 + velocity * weight#3

================================================================================================
'''

import time
import threading
import pickle

from tkinter import *
from PIL import ImageGrab
import pyautogui
#================================================================================================
#------------------ Global Variables ---------------------
#================================================================================================

w0, w1, w2, w3 = 160.0, 1.0, 0.0, 0.0;
root = Tk()
entry1, entry2, entry3 = "", "", ""
canvas = Canvas(root, width=400, height=600)
dino_color = (83, 83, 83)
MX_R, MX_C = 75, 200
GM_R, GM_C = 40, 40
#SC_R, SC_C = 14, 12 #70
SC_R, SC_C = 14, 70

avg_speed = []
Matrix = [[0 for x in range(MX_R+1)] for y in range(MX_C+1)]
game_over = [[0 for x in range(GM_R+1)] for y in range(GM_R+1)]
mt_score = [[0 for x in range(SC_R+1)] for y in range(SC_C+1)]
mt_n = [[0 for x in range(SC_R+1)] for y in range(SC_C+1)]
game_over_ref = [[0 for x in range(GM_R+1)] for y in range(GM_R+1)]

flag_restore = False
flag_numbers = False
numbers = [0 for x in range(0,10)]
#---------------------------------------------------------

def reset():
    global lp, v, dr, t1, fps, display_fps, flag_jump, enemy, txt
    dr = 150
    v, v_ct, lp, t1, fps, display_fps = 0, 0, 0, 0, 0, 0
    flag_jump = False

def GUI():
    global entry1
    button1 = Button(root, text="Set Game Over", command=save_game_over, width=10)
    button4 = Button(root, text="Set Numbers", command=save_number, width=10)
    entry1 = Entry(root, width=10, text="0")

    button2 = Button(root, text="Reset", command=reset, width=10)
    button3 = Button(root, text="Exit", command=quit, width=10)

    pdx, pdy = 5, 5
    button1.grid(row=1, column=1, padx=(pdx, pdx), pady=(pdy, pdy))
    button2.grid(row=1, column=2, padx=(pdx, pdx), pady=(pdy, pdy))
    button3.grid(row=1, column=3, padx=(pdx, pdx), pady=(pdy, pdy))
    entry1.grid(row=2, column=1, padx=(pdx, pdx), pady=(pdy, pdy))
    button4.grid(row=2, column=2, padx=(pdx, pdx), pady=(pdy, pdy))

    canvas.grid(row=3, column=1, padx=(pdx, pdx), pady=(pdy, pdy), columnspan=3)

    root.title("DinoBot's Vision - by P. Ramos - Jun 2018")
    root.geometry('%dx%d+%d+%d' % (400, 600, 1152-420, 0))

    root.after(1, events_controller)
    root.mainloop()

def events_controller():
    draw_screen()
    root.after(1, events_controller)

def load_mtx(screen, mtx, xi, yi, mx, my, ft):
    for x in range(0, mx):
        for y in range(0, my):
            color = screen.getpixel((x*ft+xi, y*ft+yi))
            if color == dino_color:
                cx = 1
            else:
                cx = 0
            mtx[x][y] = cx

last_number = 0
def load_matrix():
    global last_number, w0, w1, w2, w3
    #------------ Loading the Matrix with the Screen Data ------------
    screen = ImageGrab.grab(bbox=(0, 0, 700, 650))
    load_mtx(screen, Matrix, 25, 290, MX_C, MX_R, 2) # Matrix p1(25, 290), p2(612, 440)
    load_mtx(screen, game_over, 285, 348, GM_C, GM_R, 1)  # Game Over p1(285, 325), p2(348, 388)
    load_mtx(screen, mt_score, 536, 276, SC_C, SC_R, 1)  # Score p1(285, 325), p2(348, 388)
    #load_mtx(screen, mt_score, 573, 276, SC_C, SC_R, 1)  # Score p1(285, 325), p2(348, 388)

    if compare_game_over()==True:
        list = [4, 15, 26, 37, 48]
        dig = [0, 0, 0, 0, 0]
        ct = 0
        sc = 0
        for dxb in list:
            n = compare_numbers(dxb)
            dig[ct] = n
            ct = ct+1
        score = dig[4] + dig[3]*10 + dig[2]*100 + dig[1]*1000 + dig[0]*10000


        print('Game Over - Score ', score)
        print('fx(x) = A + Mx')
        print('w0={}, w1={}, w2={}, w3={}'.format(w0, w1, w2, w3))
        w2 += 0.01

        thread_jump();
    #-------------------------------------------------------------------

def fix_matrix():
    combinations = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for x in range(1, MX_C):
        for y in range(1, MX_R):
            cx = Matrix[x][y]
            if cx==1:
                nb = 0
                for cb in combinations:
                    dx, dy = cb
                    if Matrix[x+dx][y+dy]==1:
                        nb +=1
                if nb<=2:
                    Matrix[x][y]=0


def detect_objects():
    fix_matrix()
    DXI, DXF, DYI, DYF = 25, MX_C, 0, 50
    x1, x2 = 0, 0
    y1, y2 = 0, 0
    yr = 0
    for x in range(DXI, DXF):
        for y in range(DYI, DYF):
            cx = Matrix[x][y]
            if cx==1:
                y1, y2 = y, y
                x1 = x*3
                #--------- Calculating the Size (Width) of the Object --------------
                size = 0
                for sz in range(x, x+25):
                    for dy in range(-20, 20):
                        try:
                            csz = Matrix[sz][y+dy]
                            yr = y+dy
                        except:
                            csz = 0
                        if csz==1:
                            if y1>yr:
                                y1 = yr
                            if y2<yr:
                                y2 = yr
                            tsz = (sz-x) * 4.5
                            if size<tsz:
                                size = tsz
                if size<50:
                    size=50
                #-------------------------------------------------------------------
                return x1, x1+size, y1*3, y2*3
    return x1, x2, y1, y2


def draw_rect(tuple_xy, color, ft):
    global canvas
    x1, y1, x2, y2 = tuple_xy
    canvas.create_rectangle(x1/ft, y1, x2/ft, y2, fill=color)

def thread_jump():
    pyautogui.press("up")  # Jump

def start_fps():
    global fps, display_fps
    display_fps = fps
    fps = 0
    root.after(1000, start_fps)


def draw_dinos_vision():
    global canvas, root, enemy, txt, numbers, flag_numbers
    ft = 1.5
    dsl_y = 150
    canvas.delete(ALL)
    draw_rect(enemy, "red", ft)
    canvas.create_text(150, 150, fill="darkblue", font="Times 10", text=txt, width=300)
    # ---------------- Displaying the Screen Contents-------------------
    ct = 0
    for x in range(0, MX_C):
        for y in range(0, MX_R):
            cx = Matrix[x][y]
            tp_xy = (x * 2 + 10, y * 2 + 50+ dsl_y, x * 2 + 11, y * 2 + 51+dsl_y)
            if cx == 1:
                draw_rect(tp_xy, "blue", 1)
        ct += 1
        if (ct == 10):
            ct = 0
            x1, y1, x2, y2 = (x * 2 + 10, 50 + dsl_y, x * 2 + 10, MX_R * 2 + 52 + dsl_y)
            canvas.create_line(x1, y1, x2, y2, fill="green")
    canvas.create_line(12, 163 + dsl_y, 612, 163 + dsl_y, fill="blue")  # floor
    # -----------------------Drawin Score--------------------------------
    for x in range (0, SC_C):
        for y in range(0, SC_R):
            cx = mt_score[x][y]
            tp_xy = (x * 1 + 100, y * 1.5 + 25+ dsl_y, x * 1 + 101, y * 1.5 + 26+dsl_y)
            if cx == 1:
                draw_rect(tp_xy, "blue", 1)
    # -----------------------Drawing Validations ---------------------------
    if flag_numbers==False:
        restore_numbers()

    for n in range(0, 11):
        if n==10:
            mt_n = game_over_ref
            r, c, ft = 40, 40, 1
        else:
            mt_n = numbers[n]
            r, c, ft = 12, 14, 1.5

        for x in range(0, r):
            for y in range(0, c):
                cx = mt_n[x][y]
                tp_xy = (x * 1 + 10+ n*20, y * ft + 220 + dsl_y, x * 1 + 11+ n*20, y * ft + 221 + dsl_y)
                if cx == 1:
                    draw_rect(tp_xy, "red", 1)

    # ----------------------------------------------------------------------
    # -----------------------Drawin Score--------------------------------
    st = str(entry1.get())
    dxb =0
    if st!="":
        dxb =int(st)
    for x in range (dxb, 12+dxb):
        for y in range(0, SC_R):
            cx = mt_score[x][y]
            tp_xy = (x * 1 + 10, y * 1.5 + 260+ dsl_y, x * 1 + 11, y * 1.5 + 261+dsl_y)
            if cx == 1:
                draw_rect(tp_xy, "blue", 1)

#------------------------- Game Over Detection -----------------------------
def compare_game_over():
    global game_over_ref
    global flag_restore
    if flag_restore==False:
        restore_game_over()

    for x in range(0, 40):
        for y in range(0,40):
            if game_over[x][y]!=game_over_ref[x][y]:
                return False
    return True

def save_game_over():
    with open('game_over.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(game_over, f)
    restore_game_over()

def restore_game_over():
    global game_over_ref
    with open('game_over.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        game_over_ref = pickle.load(f)
    flag_restore == True

#--------------------------- Score Detection ------------------------------------
def save_number():
    n = int(str(entry1.get()))
    with open(str(n)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(mt_score, f)
    n +=1
    entry1.delete(0,END)
    entry1.insert(0,str(n))
    print("Number "+ str(n-1)+ " Saved!")


def restore_numbers():
    global numbers, flag_numbers
    for n in range(0,10):
        try:
            with open(str(n)+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                numbers[n] = pickle.load(f)
                print ("Number {} loaded!".format(n))
        except:
            pass
    flag_numbers=True


def compare_matrix(x, y, mt_a, mt_b):
    for x in range(0, x):
        for y in range(0, y):
            if mt_a[x][y] != mt_b[x][y]:
                return False
    return True


def compare_matrix_align(x, y, mt_a, mt_b, dxb):
    for x in range(0, x):
        for y in range(0, y):
            try:
                if mt_a[x][y] != mt_b[x+dxb][y]:
                    return False
            except:
                return False
    return True


def compare_numbers(dxb):
    global flag_numbers
    if flag_numbers==False:
        restore_numbers()
    for n in range(0, 10):
        n_mtx = numbers[n]
        if compare_matrix_align(12, 14, n_mtx, mt_score, dxb):
            return n
    return -1
#-------------------------------------------------------------------------------

def draw_screen():
    global lp, v, ct,  dr, t1, fps, flag_jump, avg_speed, enemy, txt
    global w0, w1, w2, w3
    dsl_y = 150
    load_matrix()
    x1, x2, y1, y2 = detect_objects()

    enemy = (x1, y1 + dsl_y, x2, y2 + dsl_y)
    d, s = (x1-75), (x2-x1)

    if s<0:
        s=0
    str = ""
    if flag_jump:
        str = " Jump!"
    txt = "d: {}, sz: {}, v: {}p/ms dr: {} \nfps={} cycle={}ms \n{}".format(d,s,int(v),int(dr), display_fps, int((time.time() - t1) * 1000), str)

    #------------- Draw Dino's vision ---------------
    draw_dinos_vision()
    #------------- Speed Calculation ----------------
    t = time.time()
    if lp>0 and d>0:
        if (lp - d) > 0 and t1 != 0:
            vc = int(((lp - d) / (t -t1))/10)*10
            if vc>0:
                avg_speed.append(vc)

            if (len(avg_speed)>=10):
                vt = 0
                for vi in avg_speed:
                    vt += vi
                v = vt / len(avg_speed)
                avg_speed = []

    #dr = w0 + d * w1 + v * w2 + s * w3
    dr = w0 + v/10 * (1+w2) - s/4
    lp, t1 = d, t
    #--------------- Jump ---------------------------
    flag_jump = False
    if d>0 and d <= int(dr):
        #thread_jump()
        th1 = threading.Thread(target=thread_jump)
        th1.start()
        th1.join()
        flag_jump=True
    #-----------------------------------------------
    fps +=1


reset()
start_fps()
GUI()