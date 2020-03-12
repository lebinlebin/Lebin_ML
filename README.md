# ML_Learning_code
ssh-add github_rsa 


###解决绘图乱码问题
cd /Users/liulebin/anaconda3/envs/tensorflow1.14/lib/python3.6/site-packages/matplotlib/  
/Users/liulebin/anaconda3/envs/tensorflow2.0/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf  
cp /Users/liulebin/Downloads/SimHei.ttf  ./   

    
找到font.family和font.sans-serif两项，将#注释去掉，打开。    
font-family : sans-serif    
font.sans-serif : SimHei,   
axes.unicode_minus  : False     

代码中加入   
from matplotlib.font_manager import _rebuild   
_rebuild() #reload一下  