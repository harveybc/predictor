<?php
$this->breadcrumbs=array(
	'Avisos Zis'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Avisos ZI', 'url'=>array('index')),
	array('label'=>'Gestionar Avisos ZI', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Avisos ZI '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'avisos-zi-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Create')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
