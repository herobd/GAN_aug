from data.sythetic_text import SyntheticText
import cv2

st = SyntheticText(line_prob=0.8,line_thickness=70,line_var=30,pad=50,gaus_noise=0.25,hole_prob=0.6, hole_size=400,neighbor_gap_var=40,rot=6)

for i in range(20):
    image,text= st.getSample()
    print('{}: {}'.format(i,text))
    fn = 'test/{}.png'.format(i)
    cv2.imwrite(fn,255*image)

    #fn = 'test/{}_main.png'.format(i)
    #cv2.imwrite(fn,255*maincrop)
    #fn = 'test/{}_A.png'.format(i)
    #cv2.imwrite(fn,255*Acrop)
