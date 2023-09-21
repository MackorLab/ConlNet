#!/usr/bin/env python

from __future__ import annotations

import gradio as gr
import torch

from app_canny import create_demo as create_demo_canny
from app_depth import create_demo as create_demo_depth
from app_ip2p import create_demo as create_demo_ip2p
from app_lineart import create_demo as create_demo_lineart
from app_mlsd import create_demo as create_demo_mlsd
from app_normal import create_demo as create_demo_normal
from app_openpose import create_demo as create_demo_openpose
from app_scribble import create_demo as create_demo_scribble
from app_scribble_interactive import \
    create_demo as create_demo_scribble_interactive
from app_segmentation import create_demo as create_demo_segmentation
from app_shuffle import create_demo as create_demo_shuffle
from app_softedge import create_demo as create_demo_softedge
from model import Model
from settings import (ALLOW_CHANGING_BASE_MODEL, DEFAULT_MODEL_ID,
                      SHOW_DUPLICATE_BUTTON)

DESCRIPTION = '# DIAMONIK7777 - ControlNet + Individual Model'
DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

model = Model(base_model_id=DEFAULT_MODEL_ID, task_name='Canny')

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value='Duplicate Space for private use',
                       elem_id='duplicate-button',
                       visible=SHOW_DUPLICATE_BUTTON)

    with gr.Tabs():
        with gr.TabItem('Canny'):
            create_demo_canny(model.process_canny)
        with gr.TabItem('MLSD'):
            create_demo_mlsd(model.process_mlsd)
        with gr.TabItem('Scribble'):
            create_demo_scribble(model.process_scribble)
        with gr.TabItem('Scribble Interactive'):
            create_demo_scribble_interactive(
                model.process_scribble_interactive)
        with gr.TabItem('SoftEdge'):
            create_demo_softedge(model.process_softedge)
        with gr.TabItem('OpenPose'):
            create_demo_openpose(model.process_openpose)
        with gr.TabItem('Segmentation'):
            create_demo_segmentation(model.process_segmentation)
        with gr.TabItem('Depth'):
            create_demo_depth(model.process_depth)
        with gr.TabItem('Normal map'):
            create_demo_normal(model.process_normal)
        with gr.TabItem('Lineart'):
            create_demo_lineart(model.process_lineart)
        with gr.TabItem('Content Shuffle'):
            create_demo_shuffle(model.process_shuffle)
        with gr.TabItem('Instruct Pix2Pix'):
            create_demo_ip2p(model.process_ip2p)

    with gr.Accordion(label='Base model', open=False):
        with gr.Row():
            with gr.Column(scale=5):
                current_base_model = gr.Text(label='Current base model')
            with gr.Column(scale=1):
                check_base_model_button = gr.Button('Check current base model')
        with gr.Row():
            with gr.Column(scale=5):
                new_base_model_id = gr.Text(
                    label='New base model',
                    max_lines=1,
                    placeholder='runwayml/stable-diffusion-v1-5',
                    info=
                    'The base model must be compatible with Stable Diffusion v1.5.',
                    interactive=ALLOW_CHANGING_BASE_MODEL)
            with gr.Column(scale=1):
                change_base_model_button = gr.Button(
                    'Change base model', interactive=ALLOW_CHANGING_BASE_MODEL)
        if not ALLOW_CHANGING_BASE_MODEL:
            gr.Markdown(
                '''The base model is not allowed to be changed in this Space so as not to slow down the demo, but it can be changed if you duplicate the Space.'''
            )

    check_base_model_button.click(
        fn=lambda: model.base_model_id,
        outputs=current_base_model,
        queue=False,
        api_name='check_base_model',
    )
    new_base_model_id.submit(
        fn=model.set_base_model,
        inputs=new_base_model_id,
        outputs=current_base_model,
        api_name=False,
    )
    change_base_model_button.click(
        fn=model.set_base_model,
        inputs=new_base_model_id,
        outputs=current_base_model,
        api_name=False,
    )

demo.queue(max_size=20).launch(debug=True, max_threads=True, share=True, inbrowser=True)
