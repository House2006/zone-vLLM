$$
\begin{bmatrix}
-q_{\frac{d}{2}}\\
-q_{\frac{d}{2}+1}\\
-q_{\frac{d}{2}+2}\\
-q_{\frac{d}{2}+3}\\
.\\
.\\
.\\
-q_{d-1}\\

q_0\\
q_1\\
q_2\\
q_3\\
.\\
.\\
.\\
q_{\frac{d}{2}-1}\\
\end{bmatrix}
\odot
\begin{bmatrix}
sin\theta_0\\
sin\theta_1\\
sin\theta_2\\
sin\theta_3\\
.\\
.\\
.\\
sin\theta_{\frac{d}{2}-1}\\

sin\theta_0\\
sin\theta_1\\
sin\theta_2\\
sin\theta_3\\
.\\
.\\
.\\
sin\theta_{\frac{d}{2}-1}
\end{bmatrix}
+
\begin{bmatrix}
q_0\\
q_1\\
q_2\\
q_3\\
.\\
.\\
.\\
q_{\frac{d}{2}-1}\\
q_{\frac{d}{2}}\\
q_{\frac{d}{2}+1}\\
q_{\frac{d}{2}+2}\\
q_{\frac{d}{2}+3}\\
.\\
.\\
.\\
-q_{d-1}
\end{bmatrix}
\odot
\begin{bmatrix}
cos\theta_0\\
cos\theta_1\\
cos\theta_2\\
cos\theta_3\\
.\\
.\\
.\\
cos\theta_{\frac{d}{2}-1}\\

cos\theta_0\\
cos\theta_1\\
cos\theta_2\\
cos\theta_3\\
.\\
.\\
.\\
cos\theta_{\frac{d}{2}-1}
\end{bmatrix}
=
\begin{bmatrix}
q_0\cos\theta_0-q_{\frac d2}\sin\theta_0\\
q_1\cos\theta_1-q_{\frac d2+1}\sin\theta_1\\
q_2\cos\theta_2-q_{\frac d2+2}\sin\theta_2\\
q_3\cos\theta_3-q_{\frac d2+3}\sin\theta_3\\
.\\
.\\
.\\
q_{\frac d2-1}\cos\theta_{\frac d2-1}-q_{d-1}\sin\theta_{\frac d2-1}\\
q_0\sin\theta_0+q_{\frac d2}\cos\theta_0\\
q_1\sin\theta_1+q_{\frac d2+1}\cos\theta_1\\
q_2\sin\theta_2+q_{\frac d2+2}\cos\theta_2\\
q_3\sin\theta_3+q_{\frac d2+3}\cos\theta_3\\
.\\
.\\
.\\ 
q_{\frac d2-1}\sin\theta_{\frac d2-1}-q_{d-1}\cos\theta_{\frac d2-1}
\end{bmatrix}
$$

 

$$
\begin{bmatrix}
-q_1\\
q_0\\
-q_3\\
q_2\\
.\\
.\\
.\\
-q_{d-1}\\
q_{d-2}
\end{bmatrix}
\odot
\begin{bmatrix}
\sin\theta_0\\
\sin\theta_0\\
\sin\theta_1\\
\sin\theta_1\\
.\\
.\\
.\\
\sin\theta_{\frac{d}{2}-1}\\
\sin\theta_{\frac{d}{2}-1}
\end{bmatrix}
+
\begin{bmatrix}
q_0\\
q_1\\
q_2\\
q_3\\
.\\
.\\
.\\
q_{d-2}\\
q_{d-1}
\end{bmatrix}
\odot
\begin{bmatrix}
\cos\theta_0\\
\cos\theta_0\\
\cos\theta_1\\
\cos\theta_1\\
.\\
.\\
.\\
\cos\theta_{\frac{d}{2}-1}\\
\cos\theta_{\frac{d}{2}-1}
\end{bmatrix}
=
\begin{bmatrix}
q_0\cos\theta_0-q_1\sin\theta_0\\
q_0\sin\theta_0+q_1\cos\theta_0\\
q_2\cos\theta_1-q_3\sin\theta_1\\
q_2\sin\theta_1+q_3\cos\theta_1\\
.\\
.\\
.\\
q_{d-2}\cos\theta_{\frac{d}{2}-1}-q_{d-1}\sin\theta_{\frac{d}{2}-1}\\
q_{d-2}\sin\theta_{\frac{d}{2}-1}+q_{d-1}\cos\theta_{\frac{d}{2}-1}
\end{bmatrix}
$$
