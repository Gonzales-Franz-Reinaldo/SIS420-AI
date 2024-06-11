# Juego Tetris con Agente de Aprendizaje por Refuerzo

## Introducción
Este informe documenta el desarrollo de un juego de Tetris implementado en Python, junto con un Agente de Aprendizaje por Refuerzo (RL). La combinación de estos elementos permite explorar la aplicación de técnicas de inteligencia artificial en un entorno de juego clásico. Se describen los componentes clave del juego, la lógica del agente y la interfaz de usuario. Además, se detallan los requisitos, las instrucciones para la ejecución y se presentan conclusiones sobre el proyecto.
<p align="center">
  <a href="https://github.com/dancrewzus/tetris-ia" target="blank"><img src="https://raw.githubusercontent.com/dancrewzus/tetris-ia/main/assets/tetris.jpg" width="200" alt="Tetris Image" /></a>
</p>

## Objetivos del proyecto
1. **Implementación del Juego Tetris:** Desarrollar una versión funcional del juego Tetris utilizando la biblioteca Pygame.

2. **Agente de Aprendizaje por Refuerzo:** Integrar un agente de aprendizaje por refuerzo para mejorar su rendimiento en el juego a lo largo del tiempo.

3. **Interfaz Gráfica Atractiva:** Diseñar una interfaz de usuario (UI) intuitiva y atractiva para que los usuarios interactúen con el juego.

## Requisitos
Asegúrese de tener instaladas las siguientes bibliotecas antes de ejecutar el juego:

- Python 3.x
- Pygame
- TensorFlow
- NumPy
- Pickle

Puede instalar las bibliotecas faltantes usando el siguiente comando:

```bash
$ pip install pygame tensorflow numpy
```

## Ejecución del Juego
1. Descargue el código fuente del repositorio.
2. Navegue a la carpeta del proyecto en la terminal.
3. Ejecute el siguiente comando: 
```bash
$ python main.py
```

## Estructura del Proyecto
El proyecto consta de tres partes principales:

1. **Juego Tetris (Game Logic):** Implementado en el archivo game_logic.py, se encarga de la lógica del juego, incluyendo la generación de piezas, manejo de eventos del jugador e IA, y la lógica de puntuación.

2. **Agente de Aprendizaje por Refuerzo (RLAgent):** Se encuentra en el archivo rl_agent.py. Este agente utiliza una red neuronal para aprender a tomar decisiones y maximizar su puntuación en el juego.

3. **Interfaz de Usuario (Game UI):** La interfaz gráfica del juego está en el archivo game_ui.py. Se encarga de dibujar la cuadrícula del juego, las piezas, el puntaje y otros elementos visuales.

## Funcionamiento del Juego
### **Lógica del Juego**

1. **Inicio del Juego:**

- La función '**start**' en game_ui.py muestra la pantalla de inicio.
- El juego comienza a los 3 segundos de mostrar la pantalla de bienvenida.

2. **Ejecución del Juego:**

- La función '**runGame**' controla la ejecución del juego.
- La cuadrícula se actualiza en cada fotograma, y las piezas caen a una velocidad controlada por el tiempo.
- El jugador puede mover, rotar y bajar las piezas usando las teclas del teclado.

3. **IA en Acción:**

- En paralelo con el jugador, el agente de RL toma decisiones sobre cómo mover la pieza actual.
- El agente aprende de su experiencia y ajusta su comportamiento para maximizar las recompensas.

4. **Fin del Juego:**

- El juego termina cuando no se puede colocar una nueva pieza en la pantalla.
- Se muestra una pantalla de fin de juego con la puntuación final.

### **Lógica del Agente de RL**

1. **Inicialización:**

- El agente se inicializa con una red neuronal, parámetros de aprendizaje y una memoria de experiencia.

2. **Entrenamiento Continuo:**

- Durante la ejecución del juego, el agente toma decisiones y aprende de su experiencia.
- La función replay realiza el entrenamiento utilizando la memoria de experiencia acumulada.

3. **Exploración y Explotación:**

- El agente equilibra la exploración y la explotación para mejorar su rendimiento.
- La probabilidad de exploración disminuye con el tiempo.

4. **Persistencia del Modelo:**

- El modelo del agente se guarda al finalizar el juego para su reutilización.

## Posibles Mejoras
1. **Optimización de hiperparámetros:** Se pueden realizar ajustes adicionales en los hiperparámetros del agente de RL para mejorar aún más su rendimiento.

2. **Interfaz de usuario avanzada:** Se pueden agregar elementos visuales adicionales, como efectos y animaciones, para mejorar la experiencia del usuario.

3. **Implementación de otros Algoritmos de RL:** Explorar la implementación de otros algoritmos de aprendizaje por refuerzo para comparar y contrastar su rendimiento.

## Conclusiones
El juego Tetris implementado con un agente de RL demuestra la aplicación del aprendizaje por refuerzo en entornos de juegos. El agente mejora su rendimiento a medida que acumula experiencia, ajustando su estrategia para obtener mayores recompensas.

Este proyecto sirve como ejemplo práctico de la integración de juegos, aprendizaje por refuerzo y una interfaz de usuario en Python. Puede servir como base para proyectos más complejos y exploraciones en el campo de la inteligencia artificial y los videojuegos.

Espero que este informe proporcione una comprensión detallada de la implementación y funcionamiento del juego Tetris con un agente de aprendizaje por refuerzo. 

**¡Diviértase explorando y mejorando el juego!**

## Ponte en contacto

- Desarrollador - [Daniel Rodríguez](https://www.instagram.com/dancrewzus)
- Website - [https://dancrewzus.github.io/](https://dancrewzus.github.io/)

## Licencia

[MIT licensed](LICENSE).