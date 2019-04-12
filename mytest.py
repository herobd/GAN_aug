from data.sythetic_text import SyntheticText
import cv2

st = SyntheticText('../data/text_fonts','../data/OANC_text',line_prob=0.8,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=30,rot=2.5,text_len=40)

for i in range(30):
    image,text= st.getSample()
    minV = image.min()
    maxV = image.max()
    print('{} min:{}, max:{},  text:{}'.format(i,minV,maxV,text))
    fn = 'test/{}.png'.format(i)
    cv2.imwrite(fn,255*image)

    #fn = 'test/{}_main.png'.format(i)
    #cv2.imwrite(fn,255*maincrop)
    #fn = 'test/{}_A.png'.format(i)
    #cv2.imwrite(fn,255*Acrop)
