<?php
$this->breadcrumbs=array(
	'Ubicación Técnica'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Ubicación Técnica', 'url'=>array('index')),
	array('label'=>'Gestionar Ubicación Técnica', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Ubicación Técnica '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'ubicacion-tec-form',
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
