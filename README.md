# PROJETE 2021
Monitoramento residencial utilizando rede neural desenvolvida com PyTorch no Raspberry Pi.

Video explicativo: https://youtu.be/5svkakawzIo

O arquivo utilizado para gerar a inteligência artificial (LSTM), junto com alguns PDFs gerados em testes estão na pasta "Forecasting_training"
Na pasta "raspberry_pi" o arquivo Main.py, salvo no Raspberry Pi carrega os parâmetros das inteligências artificiais (.nn), lê os gráficos de exemplo (.prjt), gera uma previsão do consumo para o próximo dia e manda para o nosso servidor.
Na pasta "raspberry_pico" a programação do microcontrolador, feita em MicroPython, lê o estado das lâmpadas, junto com a voltagem e corrente da rede elétrica e manda via USB ao Raspberry Pi.
O código do servidor usado para transmitir as informações entre o Raspberry Pi para o aplicativo está na pasta "servidor".

Direção da informação:

![diagrama_programacao](https://user-images.githubusercontent.com/94933775/143149477-9841cef6-57d2-492c-b8e0-dbdc5209c2df.png)
