<?php
$this->breadcrumbs=array(
	'DocsEnLínea'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Documentos', 'url'=>array('index')),
	array('label' => 'Crear Doc. Online', 'url' => array('/Documentos/createOnline')),
	array('label'=>'Detalles de Documento', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Documento en Línea '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'anotaciones-form',
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
