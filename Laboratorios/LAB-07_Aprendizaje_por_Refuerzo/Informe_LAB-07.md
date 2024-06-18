# LABORATORIO #N7 - SIS420 - I.A.
## Alumnos: 
## - Gonzales Franz Reinaldo 
## - Delgadillo Llanos Sebastian 

Implementación de las diferentes estrategias con el entorno [Gymnasium](https://gymnasium.farama.org/) .

Repositorios: [Github - Sebastian Delgadillo](https://github.com/sebastianDLL/SIS420_IA/tree/main/Laboratorios/Laboratorio7 ) - 
              [Github - Franz Gonzales](https://github.com/Gonzales-Franz-Reinaldo/SIS420-AI/tree/main/Laboratorios/LAB-07_Aprendizaje_por_Refuerzo)

# Balance entre exploración y explotación

## Informe sobre Técnicas de Aprendizaje por Refuerzo Aplicadas en el Entorno Taxi-v3

Este informe detalla la aplicación y los resultados de varias técnicas de aprendizaje por refuerzo (RL) en el entorno Taxi-v3, un problema clásico de RL donde un taxi debe recoger y dejar pasajeros en ubicaciones específicas dentro de una cuadrícula. Se evaluaron las siguientes técnicas: Acción por Valor, Método Incremental, Valores Iniciales Optimistas, Selección de Acciones con Intervalo de Confianza, y Algoritmos de Gradiente.



## 1. Acción por Valor

Q-learning: Se utilizó una tabla Q inicializada en cero, con un algoritmo que actualiza los valores Q en función de la recompensa inmediata y el valor Q estimado del siguiente estado.
Parámetros: Tasa de aprendizaje (learning rate) = 0.2, factor de descuento (discount factor) = 0.9, epsilon inicial para exploración = 1.0, tasa de decaimiento de epsilon = 0.001.
Resultados:

*Exploración vs. Explotación*: Se observó un desequilibrio con predominio de la exploración, lo que resultó en un bajo rendimiento en términos de recompensa acumulada.
Desempeño: La política aprendida no fue óptima debido a la alta probabilidad de exploración incluso en etapas avanzadas del entrenamiento.

### Implementación:
```bash

    # Actualiza el contador de acciones y la suma de recompensas
            action_counts[state, action] += 1
            rewards_sum[state, action] += reward

            # Actualiza la tabla Q con la nueva información obtenida (método de acción-valor)
            q_table[state, action] = rewards_sum[state, action] / action_counts[state, action]

            # Actualizar el estado para el siguiente paso
            state = new_state


```


## 2. Método Incremental

Actualización Incremental: Se empleó una fórmula de actualización incremental para los valores Q, reduciendo gradualmente la tasa de aprendizaje.
Parámetros: Tasa de aprendizaje inicial = 0.1, epsilon inicial = 1.0, tasa de decaimiento de epsilon = 0.001.
Resultados:

*Exploración vs. Explotación*: Similar a la técnica de Acción por Valor, se observó un desequilibrio con una exploración excesiva.
Desempeño: Los valores Q no convergieron a una política estable, resultando en una baja recompensa acumulada y un rendimiento subóptimo.

### Implementación:
```bash

    # Realizar la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q con la nueva información obtenida usando implementación incremental
            q_table[state, action] += learning_rate * (reward - q_table[state, action])

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

```




## 3. Valores Iniciales Optimistas

Q-learning con Valores Optimistas: La tabla Q se inicializó con valores optimistas (altos) en lugar de ceros para incentivar la explotación desde el principio.
Parámetros: Tasa de aprendizaje = 0.3, factor de descuento = 0.9, valores iniciales Q = 10, epsilon inicial = 0.1.
Resultados:

*Exploración vs. Explotación*: La técnica promovió la explotación al principio, mejorando el equilibrio entre exploración y explotación.
Desempeño: La política aprendida fue significativamente mejor que con valores iniciales en cero, mostrando una convergencia más rápida y recompensas acumuladas más altas.

### Implementación:
```bash

    # Inicializa la tabla Q con valores optimistas (por ejemplo, 10)
    q_table = np.ones((env.observation_space.n, env.action_space.n)) * 10

```




## 4. Selección de Acciones con Intervalo de Confianza

UCB (Upper Confidence Bound): Se utilizó un enfoque UCB para seleccionar acciones, balanceando la exploración y explotación mediante una fórmula que considera la incertidumbre en las estimaciones de Q.
Parámetros: Tasa de aprendizaje = 0.3, factor de descuento = 0.9, constante de exploración (c) = 2.0.
Resultados:

*Exploración vs. Explotación*: Se logró un equilibrio adecuado, permitiendo una exploración efectiva al inicio y mayor explotación posteriormente.
Desempeño: La técnica UCB produjo una política robusta con altas recompensas acumuladas y un buen rendimiento general.

### Implementación:
```bash

     t += 1

            # Implementación de UCB
            ucb_values = q_table[state, :] + c * np.sqrt(np.log(t + 1) / action_counts[state, :])
            action = np.argmax(ucb_values)

            new_state, reward, terminated, truncated, _ = env.step(action)
            action_counts[state, action] += 1

            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state

```



## 5. Algoritmos de Gradiente

Policy Gradient: Se implementó un algoritmo de ascenso por gradiente en políticas, ajustando las preferencias de acción H en lugar de los valores Q.
Parámetros: Tasa de aprendizaje = 0.1, factor de descuento = 0.9.
Resultados:

*Exploración vs. Explotación*: La técnica permitió un balance dinámico y eficiente entre exploración y explotación.
Desempeño: Los resultados fueron positivos, con una política que mejoró progresivamente y recompensas acumuladas significativas.

### Implementación:
```bash

    # Actualiza las preferencias H utilizando el algoritmo de ascenso por gradiente
        for t in range(len(rewards)):
            state = states[t]
            action = actions[t]
            action_probs = np.exp(H[state]) / np.sum(np.exp(H[state]))
            for a in range(env.action_space.n):
                if a == action:
                    H[state, a] += learning_rate * (discounted_rewards[t] - avg_reward) * (1 - action_probs[a])
                else:
                    H[state, a] -= learning_rate * (discounted_rewards[t] - avg_reward) * action_probs[a]

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = episode_reward

```



## Conclusiones
*Acción por Valor y Método Incremental*: Ambas técnicas mostraron un desequilibrio significativo entre exploración y explotación, resultando en un bajo rendimiento.

*Valores Iniciales Optimistas y UCB*: Estas técnicas lograron mejores balances, promoviendo una explotación más temprana y eficiente, con un rendimiento superior.

*Algoritmos de Gradiente*: Mostraron ser efectivos en encontrar políticas robustas y equilibradas, con buenos resultados en términos de recompensa acumulada.

