<?php
    function colorEstado($EstadoIn)
    {
        if ($EstadoIn==0) return('<img src="/images/verde.gif" height="15" width="15" /> 0 - Adecuado');
        if ($EstadoIn==1) return('<img src="/images/amarillo.gif" height="15" width="15" /> 1 - Posible deficiencia - Se requiere más información.');
        if ($EstadoIn==2) return('<img src="/images/amarillo.gif" height="15" width="15" /> 2 - Posible deficiencia - Reparar en la próxima parada disponible');
        if ($EstadoIn==3) return('<img src="/images/rojo.gif" height="15" width="15" /> 3 - Deficiencia - Reparar tán pronto como sea posible');
        if ($EstadoIn==4) return('<img src="/images/rojo.gif" height="15" width="15" /> 4 - Deficiencia Grave - Inmediatamente');
        
    }
?>
<?php
//(isset($data->Path))?($data->Path!="")?:$data->Path:"No existe":"No existe"
$model = new Termotablero('search');
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

if (isset($model))

$this->widget('zii.widgets.grid.CGridView', array(
    'id' => 'tipo-grid',
    'dataProvider' => Termotablero::model()->searchTableros($TAG),
  //  'filter' => $model,
    'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        	'Fecha',
                array(
                    'header'=>'Estado',
                    'type'=>'raw',
                    'value' => 'colorEstado($data->Estado)',
                ),
        	'TAG',
                array(// related city displayed as a link
                    'header' => 'Reporte',
                    'type' => 'raw',
                    'value' => 'sprintf(\'<a href="/index.php/%s">Descargar</a>\',is_numeric($data->Path)?("archivos/displayArchivo?id=".$data->Path):("reportes/passthru?path=".urlencode($data->Path)) )',
                ),
                'Analista',
                
                 array(
            'class' => 'CButtonColumn',
                     'template'=>$dependeT,
        ),
    ),
));
?>
<!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->