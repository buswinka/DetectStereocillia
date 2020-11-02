import src.train
import src.evaluate
from src.dataloader import MaskData, KeypointData
import src.transforms as t
import src.utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as Tk
import skimage.io as io

from torch.utils.data import DataLoader
import torchvision

import warnings
import argparse
import sys
import PySimpleGUI as sg

import os.path
def draw_figure_w_toolbar(canvas, fig):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()

    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)

    def on_key_press(event):
        canvas.TKCanvas.mpl_connect("key_press_event", on_key_press)
    return

class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]
    # t[0] in ('Home', 'Pan', 'Zoom','Save')]

    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)

def gui():
    eval = src.evaluate.evaluate()

    sg.theme('Dark Blue 3')  # please make your windows colorful
    sg.SetOptions()
    layout = [
        [sg.T('Analyze Stereocilia App')],
        [sg.Text("Image: "), sg.In(size=(70, 1), enable_events=True, key="-FOLDER-"), sg.FileBrowse(key='-FILE-')],
        [sg.B('Analyze'), sg.B('Save Analysis'), sg.B('Exit'), sg.Text(key='Error', text='', size=(70, 1))],
        [sg.T('Figure:')],
        [sg.Column(layout=[[sg.Canvas(key='fig_cv', size=(400 * 2, 400))]], background_color='#DAE0E6', pad=(0, 0))],
    ]

    window = sg.Window(title='Graph with controls', layout=layout)
    window.Finalize()
    window.Maximize()

    im = []

    while True:
        event, values = window.Read()

        if event in [None, 'Exit']:  # always,  always give a way out!
            window.Close()
            break

        elif event == '-FOLDER-':
            plt.figure(1)
            fig = plt.gcf()
            DPI = fig.get_dpi()
            im = io.imread(values['-FILE-'])
            plt.imshow(im, cmap='Greys_r')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.tight_layout()
            draw_figure_w_toolbar(window.FindElement('fig_cv').TKCanvas, fig)

        elif event == 'Analyze':
            try:
                out, masks = eval(values['-FILE-'])
                out = out.transpose((1, 2, 0))
                window.Element('Error').Update(' ')
            except (AttributeError, RuntimeError):
                window.Element('Error').Update('Error: Could not analyze image')
                continue

            plt.figure(1)
            fig = plt.gcf()
            DPI = fig.get_dpi()
            plt.imshow(out)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.tight_layout()
            draw_figure_w_toolbar(window.FindElement('fig_cv').TKCanvas, fig)

        elif event == 'Save Analysis':
            try:
                filename = values['-FILE-']
                dir = os.path.splitext(filename)[0]
                os.mkdir(dir)
            except:
                window.Element('Error').Update('Error: Cannot save file')
                continue