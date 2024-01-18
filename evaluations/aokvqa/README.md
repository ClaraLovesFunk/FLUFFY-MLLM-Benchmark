A-OK-VQA and OK-VQA have their own script for evaluation. The script was slightly modified for utility purposes, nothing changed in the way, the output is evaluated. 

aokvqa's evaluates with accuracy. for the direct answer task they do so by regarding how many times a model's 
prediction matches the proposed answers (the answer that the aokvqa authors want the most are written the most often in a list of multiple possible direct answers).