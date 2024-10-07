all: ClearScreen AsczEngine Run

ClearScreen:
	rm -f AsczEngine.exe \
	clear

AsczEngine:
	nvcc -O2 -Xcompiler -w \
		-I include/Inner \
		-I include/Graphic3D \
		-I include/Playground \
		\
		-I libraries/SFML/include \
		-L libraries/SFML/lib \
		\
		-o AsczEngine \
		\
		src/Inner/Config.cu \
		src/Inner/FpsHandle.cu \
		src/Inner/CsLogHandle.cpp \
		src/Graphic3D/Camera3D.cu \
		src/Graphic3D/Vec3D.cu \
		src/Graphic3D/Edge3D.cu \
		src/Graphic3D/Tri3D.cu \
		src/Graphic3D/Plane3D.cu \
		src/Graphic3D/Color3D.cu \
		src/Graphic3D/Render3D.cu \
		src/Graphic3D/SFMLTexture.cu \
		\
		AsczEngine.cu \
		\
		-lsfml-system \
		-lsfml-window \
		-lsfml-graphics \
		-lsfml-audio \
		-lopenal32 \
		-rdc=true \
		--expt-relaxed-constexpr

Run:
	./AsczEngine

clean:
	rm -f AsczEngine.exe

# Add <-mwindows> so when you run AsczEngine.exe
# it doesnt open a terminal
# (unless you need debugging and stuff ofc)