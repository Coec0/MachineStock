# Stock predictor

### Docker guide:
docker build -t stock-predictor:latest .
docker run stock-predictor


#Information:
Problem: Using CUDA in containers requires linux as nvidia
drivers for windows does not support docker. 
Solution: Train models locally on computer. Place weights in a folder
that the container reads from, and uses CPU mode for inference. Better anyhow
if scale to multiple computers (that might not have nvidia gpus)