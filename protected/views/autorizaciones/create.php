<?php
$this->breadcrumbs=array(
	'Autorizaciones'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'Lista de Autorizaciones', 'url'=>array('index')),
	array('label'=>'Gestionar Autorizaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Autorizaciones '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'autorizaciones-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
