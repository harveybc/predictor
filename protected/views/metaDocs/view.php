<?php
$this->breadcrumbs=array(
	'Documentos'=>array('index'),
	$model->id,
);
$sufix = "";
if (isset($_GET['query'])) {
        $sufix = "?query=" . urlencode($_GET['query']);
}
$this->menu=array(
    
    array('label'=>'Solicitar Préstamo', 'url'=>array('/Prestamos/create?id='.$model->id)),
    array('label'=>'Lista de Documentos', 'url'=>array('index')),
            array('label' => 'Subir Doc. de Motor', 'url' => array('/Documentos/createSubirMotor'.$sufix)),
            array('label' => 'Subir Doc. de Equipo', 'url' => array('/Documentos/createSubirEquipo'.$sufix)),
            array('label' => 'Subir Doc. de Tablero', 'url' => array('/Documentos/createSubirTablero'.$sufix)),
	array('label'=>'Actualizar Documento', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Documento', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto??')),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle($model->titulo); ?>
<?php 
    $fisico=1;
    if (isset($model->ruta))
        if($model->ruta>0)
        {
            echo 'Descargar archivo: <a href="/index.php/Archivos/displayArchivo?id='.$model->ruta.'">'.$model->ruta0->nombre.' - '.$model->ruta0->tamano.' Bytes </a>';
            $fisico=0;
        }
    ?>

 <?php
        $modelA=  Anotaciones::model()->findBySQL("select * from anotaciones where documento=".$model->id);
        if (isset($modelA))
        {
            echo '<h2 style="font-size:16px;color:#333333;"> Documento en Línea (
                <a href="/index.php/anotaciones/update/'.$modelA->id.'">Editar</a>
                , <a href="/index.php/anotaciones/displayArchivo?id='.$modelA->id.'">Descargar</a>): </h2>
                <br />';
        
            echo $modelA->contenido;

            $fisico=0;
        }
        if ($fisico==1)
          printf('%s', CHtml::link("Solicitar préstamo", array('prestamos/create', 'id' => $model->id)));
  ?>

    <?php 
if ($fisico==1)
{
    
    
       $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
         'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
	//	'id',
		'modulo',
		'columna',
		'fila',
            		'disponibles',
		'existencias',
		
	)));
       

}
       ?>
<fieldset style="padding: 5px;margin: 5px;border:1px;border-style:solid;color:#e0cd7c;-moz-border-radius:5px;border-radius: 5px;-webkit-border-radius: 5px;">
<br /><h2 style="font-size:18px;color:#333333;"> Tabla de Contenido: </h2>
<ul><?php foreach($model->tablasDeContenidos as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->indice." - ".$foreignobj->descripcion, array('tablasDeContenido/view', 'id' => $foreignobj->id)));

				} ?></ul>
<?php
    printf('%s', CHtml::link("Agregar tabla de contenido", array('tablasDeContenido/create', 'id' => $model->id)));
?>
</fieldset>

    
    <?php
       
           $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
         'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
	//	'id',
		'descripcion',
		'autores',
		'version',
                'tipoContenido0.descripcion',
		'fabricante0.descripcion',
		'medio0.descripcion',
		'idioma0.descripcion',

		'cerveceria0.descripcion',
		'ubicacionT0.codigoSAP',
			'numPedido',
		'numComision',	
		'EAN13',
	),
         
        
)); ?>
        </span>
<?php
if ($fisico==1)
{
 echo '<fieldset style="padding: 5px;margin: 5px;border:1px;border-style:solid;color:#e0cd7c;-moz-border-radius:5px;border-radius: 5px;-webkit-border-radius: 5px;">
<br /><h2 style="font-size:18px;color:#961C1F;"> Historial de prestamos de este documento: </h2> ?><ul>';
 
foreach($model->prestamoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->fechaPrestamo, array('prestamos/view', 'id' => $foreignobj->id)));

				} 
printf('%s', CHtml::link("Solicitar Préstamo", array('prestamos/create', 'id' => $model->id)));
}
?>
</ul><br />


