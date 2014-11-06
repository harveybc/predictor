<?php
    function colorEstado($TAG_in)
    {
        $cadSalida="No hay datos para este motor";
        $modelTMP = new Aceitesnivel1;
        $modelTMP=Aceitesnivel1::model()->findBySql("select * from aceitesnivel1 where TAG=\"" . $TAG_in . "\" order by Fecha Desc");
        if (isset($modelTMP))
        {
            $EstadoIn=$modelTMP->Estado;
            if ($EstadoIn==0) $cadSalida='<img src="/images/verde.gif" height="15" width="15" /> Bueno';
            if ($EstadoIn==1) $cadSalida='<img src="/images/amarillo.gif" height="15" width="15" /> Aceptable';
            if ($EstadoIn==2) $cadSalida='<img src="/images/rojo.gif" height="15" width="15" /> Malo';
            if ($EstadoIn==3) $cadSalida='<img src="/images/rojo.gif" height="15" width="15" /> Malo';
        }
        return($cadSalida);
    }
?>
<div style="width:78%;float:left;">
            Estado : <?php echo colorEstado($TAG); ?>
            <?php
            // arreglos para gaficar
            $arrSalidaA = array();

// obtiene una array de modelos que tienen el tag
            $data = Aceitesnivel1::model()->findAllBySql("select * from aceitesnivel1 where TAG=\"" . $TAG . "\" order by Fecha Desc");
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
foreach ($data as $foreignobj) {
                    // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
                    //calcula los valores para graficar el estado
                    if (isset($foreignobj->Estado))
                    {
                        $grVal=0;
                        if ($foreignobj->Estado==0) $grVal=2;                            
                        if ($foreignobj->Estado==1) $grVal=1;                            
                         array_push($arrSalidaA, array(strtotime($foreignobj->Fecha) * 1000, $grVal));
                    }
       
                    
               }
// hace el gráfico de IP
               ?>
            <?php
               
            $this->Widget('ext.highstock.HighstockWidget', array(
                'id' => 'grafico2',
                'options' => array(
                    'colors' => array('#6E89C7',),
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
                    ),
                    'credits' => array('enabled' => false),
                    'legend' => array('enabled' => false),
                    'theme' => 'grid',
                    'rangeSelector' => array('selected' => 6,
                    ),
                    'title' => array('text' => 'Estado de Aceite'),
                    'xAxis' => array('maxZoom' => 14 * 24 * 36000),
                    'yAxis' => array('title' => array('text' => 'Estado')),
                    'series' => array(
                        array('name' => 'Estado', 'data' => $arrSalidaA),
                    ),
                    'plotOptions' => array(
                        'series' => array(
                            'marker' => array(
                                'enabled' => true,
                            ),
                        ),
                    ),
                    ))
            );
            ?>
  </div>
<div style="width:20%;float: right;padding-top:10px;">
         <?php
            //TODO: provisional: para uso de roles de admin, ingeniero y usuario.
                $esAdmin=0;
                $esIngeniero=0;
                if (!Yii::app()->user->isGuest)
                {
                    $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="'.Yii::app()->user->name.'"');
                }
                if (isset($modeloU)) 
                {
                    $esAdmin=$modeloU->Es_administrador;
                    $esIngeniero=$modeloU->Es_analista;
                    if ($esAdmin) $esIngeniero=1; 
                }
            $dependeT="{view}";
            if ($esIngeniero)$dependeT="{view}{update}";
            if ($esAdmin)$dependeT="{view}{update}{delete}";
            
            
            $model = new Aceitesnivel1('search');
            $this->widget('zii.widgets.grid.CGridView', array(
                'id' => 'aceitesnivel1-grid',
                'dataProvider' => Aceitesnivel1::model()->searchFechas($TAG),
                // 'filter' => $model,
                'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
                'columns' => array(
                    //	'id',
                    //	'Toma',
                    //   
                    //   'relacMotores.Area',
                    'Fecha',
                    //'Estado',
                    // 'OT',
                    // 'Analista',
                    array(
                        'class' => 'CButtonColumn',
                        'template'=>$dependeT,
                    ),
                ),
            ));
            ?>

    </div>
<!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->