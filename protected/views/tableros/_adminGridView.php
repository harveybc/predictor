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



$model = new Tableros('search');
$this->widget('zii.widgets.grid.CGridView', array(
    'id' => 'tipo-grid',
    'dataProvider' => Tableros::model()->searchTableros($tableros),
   // 'filter' => $model,
    'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        	//'id',
        	'TAG',
                'Tablero',
                 array(
            'class' => 'CButtonColumn',
                     'template'=>$dependeT,
        ),
    ),
));
?>
<!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->