<?php
$this->breadcrumbs=array(
	'Reportes'=>array('index'),
	$model->id,
);
// busca el modelo del equipo en estructura
$modelTMP=Estructura::model()->findByAttributes(array('Equipo'=>$model->Equipo));
$suffixID="";
if (isset($modelTMP1))
{
    $suffixID=$modelTMP1->id;
}
$this->menu=array(
        array('label' => Yii::t('app', 'Instrucciones'), 'url' => array('/Archivos/displayArchivo?id=24')),
	array('label'=>'Lista de Reportes', 'url'=>array('index')),
	array('label'=>'Nuevo Reporte', 'url'=>array('/reportes/create?id=' . $model->Equipo)),
	array('label'=>'Gestionar Reportes', 'url'=>array('/reportes/admin?id='.$suffixID)),
);
//TODO: provisional: para uso de roles de admin, ingeniero y usuario.
$esAdmin = 0;
$esIngeniero = 0;
if (!Yii::app()->user->isGuest) {
    $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="' . Yii::app()->user->name . '"');
}
if (isset($modeloU)) {
    $esAdmin = $modeloU->Es_administrador;
    $esIngeniero = $modeloU->Es_analista;
    if ($esAdmin)
        $esIngeniero = 1;
}
if ($esIngeniero)
    array_push($this->menu, array('label' => 'Actualizar Medición', 'url' => array('update', 'id' => $model->id)));
if ($esAdmin)
    array_push($this->menu, array('label' => 'Borrar Medición', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->id), 'confirm' => 'EstÃ¡ seguro de borrar esto?')));


?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->Equipo))
        $nombre=$modelTMP->Equipo;
if (isset($modelTMP->Fecha))
        $nombre=$nombre.' ('.$modelTMP->Fecha.')';



?>

<?php $this->setPageTitle('Detalles de reporte de fuga de gases (Ultrasonido):<br/>'.$nombre.''); ?>
<?php
    function colorEstado($EstadoIn)
    {
        if ($EstadoIn==0) return('<img src="/images/verde.gif" height="15" width="15" /> 0 - Adecuado');
        if ($EstadoIn==1) return('<img src="/images/amarillo.gif" height="15" width="15" /> 1 - Posible deficiencia - Se requiere más información.');
        if ($EstadoIn==2) return('<img src="/images/amarillo.gif" height="15" width="15" /> 2 - Deficiencia - Reparar Inmediatamente');
        if ($EstadoIn==3) return('<img src="/images/rojo.gif" height="15" width="15" /> 3 - Deficiencia - Reparar Inmediatamente');
        if ($EstadoIn==4) return('<img src="/images/rojo.gif" height="15" width="15" /> 4 - Deficiencia Grave - Inmediatamente');
        
    }
?>
<div class="forms50c">
<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
            //   	'id',
		
		 'Equipo',
                 'Area',
		'Proceso',
		'OT',
                array(
                    'label'=>'Estado',
                    'value'=> colorEstado($model->Estado),
                    'htmlOptions'=>array('style'=>'width:20px;'),
                    'type'=>'raw',),
		'Analista',
		//'Fecha',
                'COSTO',
		'Gas',
		'Presion',
		'Decibeles',
              //  'Reporte',
		'Descripcion',
	//	'ZI',
                array(
                    'label'=>'ZI',
                    'value'=> number_format($model->ZI),
                    'type'=>'text',),
		'Tamano',
		'CFM',
		'Corregido',
                    array(// related city displayed as a link
            'label' => 'Path',
            'type' => 'raw',
            'value' => (isset($model->Path)) ? (is_numeric($model->Path)?  '<a href="/index.php/archivos/displayArchivo?id='.$model->Path.'">Descargar Foto</a>' :'<a href="/index.php/reportes/passthru?path='.urlencode($model->Path).'">'.$model->Path.'</a>'): "",
        ),
                
	),
)); 
?>
    </div>
<div class="forms50c">
    <div class="forms100cb">
    <?php
// si el path es alfanumérico, muestra el archivo, sino, muestra la imágen desde la bd.
if (is_numeric($model->Path))
{
 echo '<a href="/index.php/archivos/displayArchivo?id='.$model->Path.'">';
 echo '<img src="/index.php/archivos/displayArchivo?id='.$model->Path.'" style="width:100%;border-width:1px;" />';
 echo '</a>';  
}
else
{
 echo '<a href="/index.php/reportes/passthru?path='.urlencode($model->Path).'">';
 echo '<img src="/index.php/reportes/passthru?path='.urlencode($model->Path).'" style="width:100%;border-width:1px;" />';
 echo '</a>';
}
?>
        </div>
</div>