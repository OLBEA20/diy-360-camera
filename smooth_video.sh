#!/bin/sh


IN="out.mp4"

B1=971
B2=2915

W=20

ffmpeg -i "$IN" -vf "geq=cb_expr='cb(X,Y)':cr_expr='cr(X,Y)':lum_expr='lum(X,Y)+between(X,$((B1-W)),$B1)*lerp((X-$B1+$W)/$W,0,lum($B1,Y)-lum($((B1+1)),Y))+between(X,$((B2-W)),$B2)*lerp((X-$B2+$W)/$W,0,lum($B2,Y)-lum($((B2+1)),Y))',format=rgb24" -y out_blend.mp4
