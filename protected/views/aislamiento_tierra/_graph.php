<?php
    function colorEstado($TAG_in)
    {
        $cadSalida1="No hay datos para este motor";
        $modelTMP = new Aislamiento_tierra;
        $modelTMP=Aislamiento_tierra::model()->findBySql("select * from aislamiento_tierra where TAG=\"" . $TAG_in . "\" order by Fecha Desc");
        if (isset($modelTMP))
        {
            $EstadoIn=$modelTMP->Estado;
            if ($EstadoIn==0) $cadSalida1='<img src="/images/verde.gif" height="15" width="15" /> Bueno';
            if ($EstadoIn==1) $cadSalida1='<img src="/images/amarillo.gif" height="15" width="15" /> Atención';
            if ($EstadoIn==2) $cadSalida1='<img src="/images/rojo.gif" height="15" width="15" /> 2 - Malo';
            if ($EstadoIn==3) $cadSalida1='<img src="/images/rojo.gif" height="15" width="15" /> 3 - Malo';
        }
        return($cadSalida1);
    }
?>
Estado: <?php echo colorEstado($TAG); ?>
<?php

// si existen valores para graficar
if (count($arrSalidaA)>0)
    {
    
$this->Widget('ext.highstock.HighstockWidget', array(
    'id' => 'grafico1',
    'options' => array(
        'colors' => array('#6caddf', '#8bc63e', '#fec422',),
        'chart' => array(
            'selectionMarkerFill' => '#F6EAA5',
            'backgroundColor' => '#FCFAE8',
            'borderColor' => '#AC8B3A',
            'borderWidth' => 1,
            'borderRadius' => 8,
            //  'width'=>400,
            // 'height'=>300,
            
            'plotBackgroundColor' =>'#FFFFFF',
            
            'style' => array(
                'color' => '#ffffff',
                'fontSize' => '8px'
            ),
        ),
        'rangeSelector' => array(
            'selected' => 6,
            'buttons' => array(
                array('type' => 'month',
                    'count' => 1,
                    'text' => '1m'
                ),
                array('type' => 'month',
                    'count' => 3,
                    'text' => '3m'
                ),
                array('type' => 'month',
                    'count' => 6,
                    'text' => '6m'
                ),
                array('type' => 'ytd',
                    'text' => 'YTD'
                ),
                array('type' => 'year',
                    'count' => 1,
                    'text' => '1y'
                ),
                array('type' => 'all',
                    'text' => 'All'
                ),
        )),
        'colors' => array('#6caddf', '#8bc63e', '#f79727'),
        'credits' => array('enabled' => false),
        'legend' => array('enabled' => false),
        'theme' => 'grid',
        'title' => array('text' => 'Indice de Potencia de Fases A, B, C'),
        'xAxis' => array('maxZoom' => 14 * 24 * 360000),
        'yAxis' => array('title' => array('text' => 'Indice de Potencia')),
        'series' => array(
            array('name' => 'Fase A', 'data' => $arrSalidaA),
            array('name' => 'Fase B', 'data' => $arrSalidaB),
            array('name' => 'Fase C', 'data' => $arrSalidaC),
         ),
        'plotOptions'=>array(
            'series'=>array(
                'marker'=>array(
                    'enabled'=>true,
                ),
            ),
        ),
        
        ))
);
}
else
    {
    echo "No hay datos para graficar.";
    }
//echeo "FIN";
/*
  echo "Gráfico de IP por Fase";
  CController::widget('ext.highcharts.HighchartsWidget',
  array (
  'id' => 'grafico1',
  'options' => array (
  'title' => array ('text' => 'Valores medidos'),
  'xAxis' => array (
  'title' => array ('text' => 'Fecha'),
  //'categories' => $arrFechas
  'categories' => array (4, 5, 6)
  ),
  'yAxis' => array (
  'title' => array ('text' => 'Valor')
  ),
  'credits' => array ('enabled' => false),
  'legend' => array ('enabled' => false),
  'series' => array (
  //array('name' => 'A', 'data' => $arrValoresA),
  //array('name' => 'B', 'data' => $arrValoresB),
  //array('name' => 'C', 'data' => $arrValoresC)
  array ('name' => 'Valor', 'data' => array (1, 2, 3))
  )
  )
  ));
 * */
    
?>

