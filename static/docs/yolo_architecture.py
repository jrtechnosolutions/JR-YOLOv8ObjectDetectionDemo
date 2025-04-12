"""
Script to generate a YOLO architecture diagram
Run this script to create the architecture image
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

def create_architecture_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    
    # Define colors
    colors = {
        'input': '#e3f2fd',
        'backbone': '#bbdefb',
        'neck': '#90caf9',
        'head': '#64b5f6',
        'output': '#42a5f5',
        'arrow': '#1976d2',
        'text': '#0d47a1'
    }
    
    # Input image
    ax.add_patch(Rectangle((1, 7), 2, 2, facecolor=colors['input'], edgecolor='black', alpha=0.8))
    ax.text(2, 8, "Input Image\n(640×640×3)", ha='center', va='center', fontsize=12)
    
    # Backbone (CSPDarknet)
    backbone_stages = [
        {'x': 4, 'y': 7, 'w': 2, 'h': 2, 'name': "CSP\nDarknet\nStage 1"},
        {'x': 7, 'y': 7, 'w': 2, 'h': 2, 'name': "CSP\nDarknet\nStage 2"},
        {'x': 10, 'y': 7, 'w': 2, 'h': 2, 'name': "CSP\nDarknet\nStage 3"}
    ]
    
    for stage in backbone_stages:
        ax.add_patch(Rectangle((stage['x'], stage['y']), stage['w'], stage['h'], 
                               facecolor=colors['backbone'], edgecolor='black', alpha=0.8))
        ax.text(stage['x'] + stage['w']/2, stage['y'] + stage['h']/2, stage['name'], 
                ha='center', va='center', fontsize=11)
    
    # Arrows between backbone stages
    for i in range(len(backbone_stages) - 1):
        start = backbone_stages[i]
        end = backbone_stages[i+1]
        arrow = FancyArrowPatch((start['x'] + start['w'], start['y'] + start['h']/2),
                               (end['x'], end['y'] + end['h']/2),
                               connectionstyle="arc3,rad=0.0",
                               arrowstyle="-|>", color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    # Neck (PANet)
    neck_blocks = [
        {'x': 4, 'y': 4, 'w': 2, 'h': 1.5, 'name': "Feature\nAggregation 1"},
        {'x': 7, 'y': 4, 'w': 2, 'h': 1.5, 'name': "Feature\nAggregation 2"},
        {'x': 10, 'y': 4, 'w': 2, 'h': 1.5, 'name': "Feature\nAggregation 3"}
    ]
    
    for block in neck_blocks:
        ax.add_patch(Rectangle((block['x'], block['y']), block['w'], block['h'], 
                               facecolor=colors['neck'], edgecolor='black', alpha=0.8))
        ax.text(block['x'] + block['w']/2, block['y'] + block['h']/2, block['name'], 
                ha='center', va='center', fontsize=11)
    
    # Connect backbone to neck with arrows
    for i in range(len(backbone_stages)):
        start = backbone_stages[i]
        end = neck_blocks[i]
        arrow = FancyArrowPatch((start['x'] + start['w']/2, start['y']),
                               (end['x'] + end['w']/2, end['y'] + end['h']),
                               connectionstyle="arc3,rad=0.0",
                               arrowstyle="-|>", color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    # Horizontal connections in neck
    for i in range(len(neck_blocks) - 1):
        start = neck_blocks[i]
        end = neck_blocks[i+1]
        arrow1 = FancyArrowPatch((start['x'] + start['w'], start['y'] + start['h']/2),
                                (end['x'], end['y'] + end['h']/2),
                                connectionstyle="arc3,rad=0.0",
                                arrowstyle="-|>", color=colors['arrow'], linewidth=2)
        arrow2 = FancyArrowPatch((end['x'], end['y'] + start['h']/4),
                                (start['x'] + start['w'], start['y'] + start['h']/4),
                                connectionstyle="arc3,rad=0.0",
                                arrowstyle="-|>", color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
    
    # Detection heads
    head_blocks = [
        {'x': 4, 'y': 1.5, 'w': 2, 'h': 1.5, 'name': "Detection\nHead 1"},
        {'x': 7, 'y': 1.5, 'w': 2, 'h': 1.5, 'name': "Detection\nHead 2"},
        {'x': 10, 'y': 1.5, 'w': 2, 'h': 1.5, 'name': "Detection\nHead 3"}
    ]
    
    for block in head_blocks:
        ax.add_patch(Rectangle((block['x'], block['y']), block['w'], block['h'], 
                               facecolor=colors['head'], edgecolor='black', alpha=0.8))
        ax.text(block['x'] + block['w']/2, block['y'] + block['h']/2, block['name'], 
                ha='center', va='center', fontsize=11)
    
    # Connect neck to heads
    for i in range(len(neck_blocks)):
        start = neck_blocks[i]
        end = head_blocks[i]
        arrow = FancyArrowPatch((start['x'] + start['w']/2, start['y']),
                               (end['x'] + end['w']/2, end['y'] + end['h']),
                               connectionstyle="arc3,rad=0.0",
                               arrowstyle="-|>", color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    # Output predictions
    output_block = {'x': 7, 'y': 0, 'w': 2, 'h': 1, 'name': "Predictions\n(Boxes, Classes, Confidence)"}
    ax.add_patch(Rectangle((output_block['x'], output_block['y']), output_block['w'], output_block['h'], 
                           facecolor=colors['output'], edgecolor='black', alpha=0.8))
    ax.text(output_block['x'] + output_block['w']/2, output_block['y'] + output_block['h']/2, output_block['name'], 
            ha='center', va='center', fontsize=11)
    
    # Arrows from heads to output
    for block in head_blocks:
        start_x = block['x'] + block['w']/2
        start_y = block['y']
        end_x = output_block['x'] + output_block['w']/2
        end_y = output_block['y'] + output_block['h']
        
        # Calculate control points for a nice curve
        control_x = (start_x + end_x) / 2
        control_y = (start_y + end_y) / 2
        
        arrow = FancyArrowPatch((start_x, start_y),
                               (end_x, end_y),
                               connectionstyle=f"arc3,rad={0.3 if start_x < end_x else -0.3}",
                               arrowstyle="-|>", color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    # Add title
    ax.text(7, 9.5, "YOLO (You Only Look Once) Architecture", 
            ha='center', va='center', fontsize=18, weight='bold', color=colors['text'])
    
    # Add section labels
    ax.text(2.5, 8, "Input", ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])
    ax.text(2.5, 6, "Backbone", ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])
    ax.text(2.5, 4, "Neck", ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])
    ax.text(2.5, 2, "Head", ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])
    ax.text(2.5, 0.5, "Output", ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])
    
    # Set limits and remove axes
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'yolo_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to {os.path.join(output_dir, 'yolo_architecture.png')}")

if __name__ == "__main__":
    create_architecture_diagram()
