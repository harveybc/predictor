    <?php


$model = new Aislamiento_tierra('search');

echo '<div class="forms100c">';


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

$this->widget('zii.widgets.grid.CGridView', array(
    'id' => 'aislamiento-tierra-grid',
    'dataProvider' => Aislamiento_tierra::model()->searchFechas($TAG),
   // 'filter' => $model,
    'cssFile' => '/themes/gridview/styles.css',
    'template'=> '{items}{pager}{summary}',
    'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        //	'id',
        //	'Toma',
        //   'TAG',
        //   'relacMotores.Area',
        'Fecha',
        
        array(
            'class' => 'CButtonColumn',
            'template'=>$dependeT,
        ),
    ),
));
echo '</div>';
?>
<!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->