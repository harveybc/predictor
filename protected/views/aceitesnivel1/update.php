<?php
$this->breadcrumbs=array(
	'Lubricantes'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Lubricantes', 'url'=>array('index')),
	array('label'=>'Crear Medición', 'url'=>array('create')),
	array('label'=>'Ver Medición', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Lubricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Lubricantes  '); ?>
<div class="form"><style>    .forms50cr{  float:left;   }</style>
<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'aceitesnivel1-form',
	'enableAjaxValidation'=>true,
)); 
echo '<div>';

echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); 
    echo '</div>';
?>
<div class="row buttons forms100c" styler="text-align: center;">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
