import src.train
import src.evaluate
import src.save
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
        [sg.B('Analyze'), sg.B('Save Analysis'), sg.B('Exit'), sg.Text(key='Error', text='', size=(50, 1))],
        [sg.T('Figure:')],
        [sg.Column(layout=[[sg.Canvas(key='fig_cv', size=(800, 400))]], background_color='#DAE0E6', pad=(0, 0))],
    ]

    window = sg.Window(title='Graph with controls', layout=layout)
    window.Finalize()
    window.Maximize()

    im = []

    _ANALYZED = False

    while True:
        event, values = window.Read()

        if event in [None, 'Exit']:  # always,  always give a way out!
            window.Close()
            break

        elif event == '-FOLDER-':
            fig, ax = plt.subplots(1)
            fig = plt.gcf()
            DPI = fig.get_dpi()
            im = io.imread(values['-FILE-'])
            ax.imshow(im, cmap='Greys_r')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.tight_layout()
            draw_figure_w_toolbar(window.FindElement('fig_cv').TKCanvas, fig)

            _ANALYZED = False

        elif event == 'Analyze':
            try:
                out, masks, keypoints, boxes = eval(values['-FILE-'])
                out = out.transpose((1, 2, 0))
                window.Element('Error').Update(' ')
                _ANALYZED = True
            except (AttributeError, RuntimeError):
                window.Element('Error').Update('Error: Could not load mask')
                _ANALYZED = False
                continue

            fig, ax = plt.subplots(1)
            DPI = fig.get_dpi()
            ax.imshow(out)

            for i in range(keypoints['keypoints'].shape[0]):
                if keypoints['scores'][i] < 0.5:
                    continue
                x = keypoints['keypoints'][i, :, 0]
                y = keypoints['keypoints'][i, :, 1]
                plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), 'b-', alpha=0.5)
                plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), 'b.', alpha=0.5)

            for i, box in enumerate(boxes['boxes']):
                if boxes['scores'][i] < 0.5:
                    continue
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                l = boxes['labels'][i].detach().cpu().numpy()
                plt.plot([x1, x2], [y2, y2], c='C' + str(l), lw=1)
                plt.plot([x1, x2], [y1, y1], c='C' + str(l), lw=1)
                plt.plot([x1, x1], [y1, y2], c='C' + str(l), lw=1)
                plt.plot([x2, x2], [y1, y2], c='C' + str(l), lw=1)

            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.tight_layout()
            draw_figure_w_toolbar(window.FindElement('fig_cv').TKCanvas, fig)

        elif event == 'Save Analysis':
            if _ANALYZED:
                filename = values['-FILE-']
                src.save.save_mask(masks, filename)
                src.save.save_boxes(boxes, filename)
            else:
                window.Element('Error').Update('Error: File has not been analyzed. Nothing to Save')
                continue