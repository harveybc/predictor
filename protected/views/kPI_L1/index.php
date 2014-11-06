<?php

$this->breadcrumbs = array(
    'ScoreboardL1'
);
?>

<?php $this->setPageTitle('Scoreboard Línea 1'); ?>

<?php

// guarda en $TAG_in el TAG del modelo actual

echo "<br/>";
// arreglos para gaficar
$arrSalidaVibLA = array();
// obtiene una array de modelos que tienen el tag= al del modelo actual
$data = KPI_L2::model()->findAllBySQL("select Eff_Shift,Fecha from KPI_L2 order by id desc limit 1");
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh



//print_r($arrSalidaTemperatura);
// hace el gráfico de IP
$this->Widget('ext.highstock.HighstockWidget', array(
    'id' => 'grafico8',
    'options' => array(
        'colors' => array('#F42539'),
        'credits' => array('enabled' => false),
        'legend' => array('enabled' => false),
        'theme' => 'grid',
        'rangeSelector' => array(
            'selected' => 1,
            'buttons' => array(
                array('type' => 'month',
                    'count' => 1,
                    'text' => '1m'
                ),
                array('type' => 'all',
                    'text' => 'All'
                ),
        )),
        'chart' => array(
            'selectionMarkerFill' => '#F6EAA5',
            'backgroundColor' => '#FCFAE8',
            'borderColor' => '#AC8B3A',
            'borderWidth' => 1,
            'borderRadius' => 8,
            //  'width'=>400,
            // 'height'=>300,
            'plotBackgroundColor' => '#FFFFFF',
            'style' => array(
                'color' => '#ffffff',
                'fontSize' => '8px'
            ),
            'events' => array(
                'load' => 'js:function() {

					// set up the updating of the chart each second
					var series = this.series[0];
					setInterval(function() {
						var x = (new Date()).getTime(), // current time
						y = Math.round(Math.random() * 100);
						series.addPoint([x, y], true, true);
					}, 3000);
                                    } ',
            ),
        ),
        'title' => array('text' => 'Vibr. L. Bomba o Reductor'),
        'xAxis' => array('maxZoom' => 100,'type'=>'datetime','tickPixelInterval'=>150),
        'yAxis' => array('title' => array('text' => 'Indice de Potencia')),
        'series' => array(
            array('name' => 'Eff_Shift ', 'data' => $arrSalidaVibLA)
        ),
        'plotOptions' => array(
            'series' => array(
            ),
        ),
        ))
);
?>

<script    >
   
    </script>