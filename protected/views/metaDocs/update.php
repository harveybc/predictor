<?php
$this->breadcrumbs=array(
	'Documentos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);
$sufix = "";
if (isset($_GET['query'])) {
        $sufix = "?query=" . urlencode($_GET['query']);
}
$this->menu=array(
	array('label'=>'Lista de Documentos', 'url'=>array('index')),
	            array('label' => 'Subir Doc. de Motor', 'url' => array('/Documentos/createSubirMotor'.$sufix)),
            array('label' => 'Subir Doc. de Equipo', 'url' => array('/Documentos/createSubirEquipo'.$sufix)),
            array('label' => 'Subir Doc. de Tablero', 'url' => array('/Documentos/createSubirTablero'.$sufix)),
	array('label'=>'Detalles Documento', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle(' Actualizar Documentos: <?php echo $model->titulo; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'meta-docs-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
