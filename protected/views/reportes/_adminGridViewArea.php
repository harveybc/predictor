
<?php
// compone la cadena de consulta
    $consultaSQL = 'SELECT SUM(COSTO) AS sumatoria FROM reportes WHERE (Area="' . $area . '")';
// se prepara el comando de SQL como en http://www.yiiframework.com/doc/guide/1.1/en/database.dao
    $command = Yii::app()->db->createCommand($consultaSQL);
// se ejecuta la consulta y los resultados quedan en un arreglo de resultados $resultados[0] es el primero
    $resultados = $command->queryAll();
// imprime todo el arreglo de resultados
//print_r($resultados);
// ejemplo de uso de un campo de uno de los resultados (Equipo del resultado 0)
    //       echo "<br/>";
//        echo '<b>Expr1='.$resultados[0]['Expr1'].'</b>'
//echo 'hola='.$resultados[0]['Expr1'];
?>


<b>Costo Total de reportes en el proceso seleccionado: </b> $<?php echo number_format($resultados[0]['sumatoria']); ?>
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

    $this->widget('zii.widgets.grid.CGridView',
                  array (
        'id' => 'reportes-grid',
        'dataProvider' => Reportes::model()->searchEquiposArea($area),
        //'filter' => $model,
        'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
        'columns' => array (
           array(
            'header'=>'Imagen',
            'type'=>'raw',
            'value' => 'sprintf(\'<img src="%s" style="height:120px;width:120px;border-width:1px;" />\',$data->Path)'
        ),
        array(
            'header'=>'Estado',
            'type'=>'raw',
            'htmlOptions'=>array('style'=>'width:20px;'),
            'value' => 'colorEstado($data->Estado)',
        ),
        'Presion',
        'Decibeles',
        'Descripcion',
        'COSTO',
        array(
            'class' => 'CButtonColumn',
            'template'=>$dependeT,
        ),
        ),
    ));
?>
