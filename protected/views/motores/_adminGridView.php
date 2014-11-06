<?php

// función que retorna el fallback gif si no existe la imágen del parámetro path. 
function getPathFoto($file)
{
        $fallback = 'c:\\wamp\\www\\images\\fallback.gif';
        // determina el tipo de imágen
        $ext = array_pop(explode('.', $file));
        $image = array_pop(explode('\\', $file));
        // tipos de imágenes permitidos
        $allowed['gif'] = 'image/gif';
        $allowed['png'] = 'image/png';
        $allowed['jpg'] = 'image/jpeg';
        $allowed['jpeg'] = 'image/jpeg';
        $allowed['pdf'] = 'application/pdf';
        $allowed['docx'] = 'application/docx';
        $allowed['doc'] = 'application/doc';        
        if (file_exists($file) && $ext != '' && isset($allowed[strToLower($ext)])) {
            $type = $allowed[strToLower($ext)];
        } else {
            $file = $fallback;
            $type = 'image/gif';
        }
        return($file);
}

$model = new Motores('search');
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
    'id' => 'motores_grid',
    'dataProvider' => Motores::model()->searchEquipos($area, $equipo),
  //  'filter' => $model,
    'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        'Codigo',
        'TAG',
        //   'Proceso',
        //   'Area',
        //  'Equipo',
        'Motor',
        'Marca',
        array(
            'header'=>'Imagen',
            'type'=>'raw',
            'value' => 'sprintf(\'<a href="/index.php/%s"><img src="/index.php/%s" style="height:120px;width:120px;border-width:1px;" /></a>\',is_numeric($data->PathFoto)?("archivos/displayArchivo?id=".$data->PathFoto):("reportes/passthru?path=".urlencode($data->PathFoto)), is_numeric($data->PathFoto)?("archivos/displayArchivo?id=".$data->PathFoto):("reportes/passthru?path=".urlencode($data->PathFoto))  )',
        ),
        /*
         * 
          'kW',
          'Rod_LC',
          'Rod_LA',
          'IP',
          'Frame',
          'Lubricante',
          'Velocidad',



          'PathFoto',
         */
        array(
            'class' => 'CButtonColumn',
            'template'=>$dependeT,
        ),
    ),
));
?>
