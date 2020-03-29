import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches

class Click():
    def __init__(self, ax, cv, button=1):
        self.cv = cv
        self.ax=ax
        self.button=button
        self.press=False
        self.move = False
        self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.c4=self.ax.figure.canvas.mpl_connect('scroll_event',self.onzoom)

    def onzoom(self, event):
        cv = self.cv
        lowest_dist = 1000
        lowest_index = -1
        for i in range(0, cv.shape[0]):
            dist = np.sqrt((cv[i,0]-event.xdata)**2 + (cv[i,1]-event.ydata)**2)
            if dist < lowest_dist:
                lowest_dist = dist
                lowest_index = i
        if lowest_dist < 2:
            if event.button == "up":
                cv[lowest_index, 2] += 0.5
            elif event.button == "down":
                if cv[lowest_index, 2] > 1.5:
                    cv[lowest_index, 2] -= 0.5
            # if event.button == "up":
            #     cv[:, 2] += 0.5
            # elif event.button == "down":
            #     cv[:, 2] -= 0.5

            print(cv)

    def onclick(self,event):
        if event.inaxes == self.ax:
            cv = self.cv
            lowest_dist = 1000
            lowest_index = -1
            for i in range(0, cv.shape[0]):
                dist = np.sqrt((cv[i,0]-event.xdata)**2 + (cv[i,1]-event.ydata)**2)
                if dist < lowest_dist:
                    lowest_dist = dist
                    lowest_index = i
            if event.button == self.button:
                if lowest_dist > 1:
                    cv = np.vstack((cv,np.array([event.xdata, event.ydata, 4., 1.])))
            if event.button == 3:

                if lowest_dist < 1:
                    cv = np.delete(cv, (lowest_index), axis=0)

            self.cv = cv

    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True

            # Find the closest point to the click
            cv = self.cv
            lowest_dist = 1000
            lowest_index = -1
            for i in range(0, cv.shape[0]):
                dist = np.sqrt((cv[i,0]-event.xdata)**2 + (cv[i,1]-event.ydata)**2)
                if dist < lowest_dist:
                    lowest_dist = dist
                    lowest_index = i
            if lowest_dist < 2:
                cv[lowest_index, 0] = event.xdata
                cv[lowest_index, 1] = event.ydata

            print(cv)


    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        else:
            global released
            released = True
        self.press=False; self.move=False