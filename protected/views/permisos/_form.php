<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
			</div>

	<div class="row">
			</div>

	<div class="row">
			</div>

	<div class="row">
			</div>


<label for="Modulos">Este permiso pertenece a los siguientes módulos</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'modulo0',
							'fields' => 'descripcion',
							'allowEmpty' => true,
							'style' => 'dropdownlist',
							)
						); ?>
			<label for="Usuarios">Este permiso fué autorizado a los siguientes usuarios</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'usuario0',
							'fields' => 'Username',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
							)
						); ?>
			<label for="Operaciones">Las operaciones que pueden realizarse con este permiso son las siguientes</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'operacion0',
							'fields' => 'descripcion',
							'allowEmpty' => true,
							'style' => 'dropdownlist',
							)
						); ?>
			<label for="Documentos">Este permiso fué asignado a los siguientes documentos</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'documento0',
							'fields' => 'descripcion',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
							)
						); ?>
			